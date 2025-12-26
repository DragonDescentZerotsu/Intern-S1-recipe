#!/usr/bin/env python3
# inspect_sharegpt_llamafactory.py

import argparse
import json
from typing import Any, Dict, Optional

from datasets import load_dataset
from transformers import AutoTokenizer

# --------- LLaMA-Factory imports (兼容不同版本路径) ----------
try:
    from llamafactory.hparams import DataArguments
except Exception:
    # 老版本可能在别的地方
    from llamafactory.hparams.data_args import DataArguments  # type: ignore

try:
    from llamafactory.data.parser import DatasetAttr
except Exception:
    from llamafactory.data.parser import DatasetAttr  # type: ignore

try:
    from llamafactory.data.converter import SharegptDatasetConverter
except Exception:
    # 有的版本叫 converters
    from llamafactory.data.converters import SharegptDatasetConverter  # type: ignore

try:
    from llamafactory.data import get_template_and_fix_tokenizer
except Exception:
    from llamafactory.data.template import get_template_and_fix_tokenizer  # type: ignore

try:
    from llamafactory.extras.constants import IGNORE_INDEX
except Exception:
    IGNORE_INDEX = -100  # fallback


def infer_sharegpt_tags(example: Dict[str, Any], messages_col: str) -> Dict[str, str]:
    """
    你的 conversations 里可能是:
      - {"from": "...", "value": "..."} (sharegpt 默认)
      - {"role": "...", "content": "..."} (你自己 normalize 过的)
    这里自动推断 role_tag/content_tag。
    """
    conv = example[messages_col]
    if not conv:
        return {"role_tag": "from", "content_tag": "value"}

    first = conv[0]
    if "from" in first and "value" in first:
        return {"role_tag": "from", "content_tag": "value"}
    if "role" in first and "content" in first:
        return {"role_tag": "role", "content_tag": "content"}

    # 兜底
    return {"role_tag": "from", "content_tag": "value"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ShareGPT jsonl/json 文件路径")
    ap.add_argument("--model", required=True, help="tokenizer 对应的模型，如 internlm/Intern-S1-mini")
    ap.add_argument("--template", required=True, help="LLaMA-Factory template 名称（你注册的）")
    ap.add_argument("--messages-col", default="conversations", help="默认 ShareGPT 的列名 conversations")
    ap.add_argument("--tools-col", default="tools", help="工具 schema 列名（可选）")
    ap.add_argument("--system-col", default="system", help="system 列名（可选）")
    ap.add_argument("--idx", type=int, default=5, help="查看第几个样本")
    ap.add_argument("--cutoff-len", type=int, default=8192)
    args = ap.parse_args()

    # 1) 读数据
    ds = load_dataset("json", data_files=args.data, split="train")
    ex = ds[args.idx]

    # 2) 构造 DatasetAttr（告诉 LLaMA-Factory：这是 sharegpt 格式，列名/tag 名是什么）
    dataset_attr = DatasetAttr(load_from="file", dataset_name="local_sharegpt")
    dataset_attr.formatting = "sharegpt"
    dataset_attr.messages = args.messages_col
    dataset_attr.tools = args.tools_col if args.tools_col in ex else None
    dataset_attr.system = args.system_col if args.system_col in ex else None

    tags = infer_sharegpt_tags(ex, args.messages_col)
    dataset_attr.role_tag = tags["role_tag"]
    dataset_attr.content_tag = tags["content_tag"]

    # 你若用 sharegpt 默认 human/gpt/observation/function_call：
    dataset_attr.user_tag = "human"
    dataset_attr.assistant_tag = "gpt"
    dataset_attr.observation_tag = "observation"
    dataset_attr.function_tag = "function_call"
    dataset_attr.system_tag = "system"

    # 3) ShareGPT → 对齐成 _prompt/_response/_system/_tools
    #    这一步就是 LLaMA-Factory 真正在训练前做的格式归一化之一:contentReference[oaicite:3]{index=3}
    data_args = DataArguments(template=args.template)
    converter = SharegptDatasetConverter(dataset_attr, data_args)
    aligned = converter(ex)

    prompt_msgs = aligned.get("_prompt", [])
    resp_msgs = aligned.get("_response", [])
    system = aligned.get("_system", None)
    tools = aligned.get("_tools", None)

    print("\n========== [ALIGNED KEYS] ==========")
    for k in ["_system", "_tools", "_prompt", "_response"]:
        v = aligned.get(k, None)
        if k in ("_prompt", "_response"):
            print(f"{k}: {len(v) if v else 0} messages")
        else:
            print(f"{k}: {type(v).__name__}  {'(None)' if v is None else ''}")

    # tools 如果是 dict/list，转成 str（LLaMA-Factory encode 接口的 tools 参数是 Optional[str]）:contentReference[oaicite:4]{index=4}
    if isinstance(tools, (dict, list)):
        tools = json.dumps(tools, ensure_ascii=False, indent=2)

    # 4) tokenizer + template
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # 不同版本 get_template_and_fix_tokenizer 的入参可能是 DataArguments 或 template 名字，做个兼容
    try:
        template_obj = get_template_and_fix_tokenizer(tok, data_args)
    except TypeError:
        template_obj = get_template_and_fix_tokenizer(tok, args.template)

    # 5) encode + decode，打印每一轮 (prompt, answer)
    all_msgs = prompt_msgs + resp_msgs
    pairs = template_obj.encode_multiturn(
        tok,
        all_msgs,
        system=system,
        tools=tools,
        # cutoff_len=args.cutoff_len,
        # reserved_label_len=1,
    )

    print("\n========== [ENCODED TURNS] ==========")
    for i, (p_ids, a_ids) in enumerate(pairs):
        p_txt = tok.decode(p_ids, skip_special_tokens=False)
        a_txt = tok.decode(a_ids, skip_special_tokens=False)

        # 训练时通常 labels 会把 prompt 部分 mask 掉（IGNORE_INDEX），只对 answer 计算 loss
        labels = [IGNORE_INDEX] * len(p_ids) + a_ids

        print(f"\n--- TURN {i} ---")
        print(f"prompt_ids: {len(p_ids)} | answer_ids: {len(a_ids)} | labels: {len(labels)}")
        print("\n[PROMPT TEXT]\n", p_txt)
        print("\n[ANSWER TEXT]\n", a_txt)

    print("\n✅ 如果你的 observation/tool 返回内容被正确渲染，你会在 [PROMPT TEXT] 里看到对应的 tool/observation role 段落。")


if __name__ == "__main__":
    main()
