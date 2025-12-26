#!/usr/bin/env python3
# inspect_alpaca_llamafactory.py

import argparse
import json
from typing import Any, Dict, Optional, List, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer

# --------- LLaMA-Factory imports (兼容不同版本路径) ----------
try:
    from llamafactory.hparams import DataArguments
except Exception:
    from llamafactory.hparams.data_args import DataArguments  # type: ignore

try:
    from llamafactory.data.parser import DatasetAttr
except Exception:
    from llamafactory.data.parser import DatasetAttr  # type: ignore

# AlpacaDatasetConverter 在不同版本可能位于不同路径
try:
    from llamafactory.data.converter import AlpacaDatasetConverter
except Exception:
    from llamafactory.data.converters import AlpacaDatasetConverter  # type: ignore

try:
    from llamafactory.data import get_template_and_fix_tokenizer
except Exception:
    from llamafactory.data.template import get_template_and_fix_tokenizer  # type: ignore

try:
    from llamafactory.extras.constants import IGNORE_INDEX
except Exception:
    IGNORE_INDEX = -100  # fallback


def _exists_nonempty(ex: Dict[str, Any], k: Optional[str]) -> bool:
    return bool(k) and (k in ex) and (ex[k] is not None) and (ex[k] != "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Alpaca jsonl/json 文件路径")
    ap.add_argument("--model", required=True, help="tokenizer 对应的模型，如 internlm/Intern-S1-mini")
    ap.add_argument("--template", required=True, help="LLaMA-Factory template 名称（建议用 empty 来观察最少壳）")
    ap.add_argument("--idx", type=int, default=0, help="查看第几个样本")

    # Alpaca 列名（允许你自定义）
    ap.add_argument("--instruction-col", default="instruction", help="instruction 列名")
    ap.add_argument("--input-col", default="input", help="input 列名（可选）")
    ap.add_argument("--output-col", default="output", help="output 列名")
    ap.add_argument("--history-col", default="history", help="history 列名（可选，形如 [[u,a],[u,a],...]）")
    ap.add_argument("--system-col", default="system", help="system 列名（可选）")
    ap.add_argument("--tools-col", default="tools", help="tools schema 列名（可选）")

    ap.add_argument("--cutoff-len", type=int, default=8192, help="可选：encode 截断长度（默认不传）")
    ap.add_argument("--show-raw", action="store_true", help="打印原始样本 JSON（可能很长）")
    ap.add_argument("--show-aligned", action="store_true", help="打印对齐后的 _prompt/_response 列表结构")
    args = ap.parse_args()

    # 1) 读数据
    ds = load_dataset("json", data_files=args.data, split="train")
    ex = ds[args.idx]

    # 简单校验必需列
    if args.instruction_col not in ex:
        raise KeyError(f"Missing instruction column: {args.instruction_col}. Available keys: {list(ex.keys())}")
    if args.output_col not in ex:
        raise KeyError(f"Missing output column: {args.output_col}. Available keys: {list(ex.keys())}")

    if args.show_raw:
        print("\n========== [RAW EXAMPLE] ==========")
        print(json.dumps(ex, ensure_ascii=False, indent=2))

    # 2) 构造 DatasetAttr（告诉 LLaMA-Factory：这是 alpaca 格式，列名是什么）
    dataset_attr = DatasetAttr(load_from="file", dataset_name="local_alpaca")
    dataset_attr.formatting = "alpaca"

    # AlpacaDatasetConverter 里：
    # - dataset_attr.prompt -> instruction
    # - dataset_attr.query  -> input
    # - dataset_attr.response -> output
    # - dataset_attr.history -> history (list of [old_prompt, old_response])
    dataset_attr.prompt = args.instruction_col
    dataset_attr.query = args.input_col if args.input_col in ex else None
    dataset_attr.response = args.output_col
    dataset_attr.history = args.history_col if args.history_col in ex else None

    dataset_attr.system = args.system_col if args.system_col in ex else None
    dataset_attr.tools = args.tools_col if args.tools_col in ex else None

    # 3) Alpaca → 对齐成 _prompt/_response/_system/_tools
    data_args = DataArguments(template=args.template)
    converter = AlpacaDatasetConverter(dataset_attr, data_args)
    aligned = converter(ex)

    prompt_msgs = aligned.get("_prompt", [])
    resp_msgs = aligned.get("_response", [])
    system = aligned.get("_system", "")
    tools = aligned.get("_tools", "")

    print("\n========== [ALIGNED KEYS] ==========")
    for k in ["_system", "_tools", "_prompt", "_response"]:
        v = aligned.get(k, None)
        if k in ("_prompt", "_response"):
            print(f"{k}: {len(v) if v else 0} messages")
        else:
            print(f"{k}: {type(v).__name__}  {'(None)' if v is None else ''}")

    if args.show_aligned:
        print("\n========== [ALIGNED _prompt/_response STRUCTURE] ==========")
        print("[_system]\n", system)
        if tools:
            print("\n[_tools]\n", tools if isinstance(tools, str) else json.dumps(tools, ensure_ascii=False, indent=2))
        print("\n[_prompt]\n", json.dumps(prompt_msgs, ensure_ascii=False, indent=2))
        print("\n[_response]\n", json.dumps(resp_msgs, ensure_ascii=False, indent=2))

    # tools 如果是 dict/list，转成 str（encode_multiturn 的 tools 参数通常是 Optional[str]）
    if isinstance(tools, (dict, list)):
        tools = json.dumps(tools, ensure_ascii=False, indent=2)

    # 4) tokenizer + template
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 不同版本 get_template_and_fix_tokenizer 的入参可能是 DataArguments 或 template 名字
    try:
        template_obj = get_template_and_fix_tokenizer(tok, data_args)
    except TypeError:
        template_obj = get_template_and_fix_tokenizer(tok, args.template)

    # 5) encode + decode，打印每一轮 (prompt, answer)
    all_msgs = prompt_msgs + resp_msgs

    encode_kwargs = dict(
        tokenizer=tok,
        messages=all_msgs,
        system=system,
        tools=tools,
    )

    # 某些版本 encode_multiturn 的签名是 encode_multiturn(tok, messages, system=..., tools=..., cutoff_len=...)
    # 这里用 try/except 兼容一下
    try:
        pairs = template_obj.encode_multiturn(
            tok,
            all_msgs,
            system=system,
            tools=tools,
            cutoff_len=args.cutoff_len,
        )
    except TypeError:
        pairs = template_obj.encode_multiturn(
            tok,
            all_msgs,
            system=system,
            tools=tools,
        )

    print("\n========== [ENCODED TURNS] ==========")
    for i, (p_ids, a_ids) in enumerate(pairs):
        p_txt = tok.decode(p_ids, skip_special_tokens=False)
        a_txt = tok.decode(a_ids, skip_special_tokens=False)

        labels = [IGNORE_INDEX] * len(p_ids) + a_ids  # 典型 SFT：prompt mask，只训 answer

        print(f"\n--- TURN {i} ---")
        print(f"prompt_ids: {len(p_ids)} | answer_ids: {len(a_ids)} | labels: {len(labels)}")
        print("\n[PROMPT TEXT]\n", p_txt)
        print("\n[ANSWER TEXT]\n", a_txt)

    print("\n✅ 你可以用 template=empty 来检查：LLaMA-Factory 是否还“额外加壳”。"
          "\n   如果你把 <|im_start|>assistant / <think>... 放在 instruction 里，应该只会出现在 [PROMPT TEXT]；"
          "\n   放在 output 里，则会出现在 [ANSWER TEXT]。")


if __name__ == "__main__":
    main()
