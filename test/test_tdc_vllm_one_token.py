import os, multiprocessing as mp

# 1) 强制 vLLM 用 spawn，而不是默认的 fork
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # vLLM 官方建议
# 如果你正在使用 vLLM v1（从堆栈看是 vllm.v1），可显式开启 v1 逻辑（多数版本默认已是 v1）
os.environ.setdefault("VLLM_USE_V1", "1")

# 2) 在 Python 侧把多进程 start method 切成 spawn（双保险）
mp.set_start_method("spawn", force=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # TODO: device GPU #

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from math import isfinite
import numpy as np
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Test TDC tasks with vLLM using preprocessed data')

    parser.add_argument('--model-path', type=str,
                        default='internlm/Intern-S1-mini',  # "checkpoints/Intern-S1-mini/full/sft-distill_DeepSeek_V32-TDC-train-set_less_save_interval/checkpoint-3000"
                        help='Path to the model checkpoint')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA adapter')
    parser.add_argument('--lora-path', type=str,
                        default="checkpoints/Intern-S1-mini/lora/sft/checkpoint-180000",
                        help='Path to LoRA adapter')
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / "DataPrepare/TDC_test_prompts_label_random",
                        help='Directory containing preprocessed test data')
    parser.add_argument('--task-groups', nargs='+',
                        default=['Tox'],
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'all'],
                        help='Task groups to run')
    parser.add_argument('--max-model-len', type=int, default=1024 * 2, help='Max model length')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92, help='GPU memory utilization')
    parser.add_argument('--max-num-seqs', type=int, default=256, help='Max number of sequences')
    parser.add_argument('--max-logprobs', type=int, default=1024, help='Max logprobs to return')
    parser.add_argument('--device', type=str, default='2', help='CUDA device ID')
    parser.add_argument('--strip-smiles-tags', action='store_false',
                        help='Remove <SMILES> and </SMILES> from prompts before inference')

    args = parser.parse_args()
    return args


def load_tasks_map(data_dir):
    """
    Maps task groups to specific files in the data directory.
    """
    mapping = {
        'Tox': [
            'Skin_Reaction.jsonl',
            'hERG.jsonl',
            'DILI.jsonl',
            'ClinTox.jsonl',
            'AMES.jsonl',
            'Tox21.jsonl',
            'ToxCast.jsonl',
            'herg_central_hERG_inhib.jsonl'
        ],
        'ADME': [
            'PAMPA_NCATS.jsonl',
            'HIA_Hou.jsonl',
            'BBB_Martins.jsonl',
            'Pgp_Broccatelli.jsonl',
            'Bioavailability_Ma.jsonl',
            'CYP2C9_Substrate_CarbonMangels.jsonl',
            'CYP2D6_Substrate_CarbonMangels.jsonl',
            'CYP3A4_Substrate_CarbonMangels.jsonl',
            'CYP1A2_Veith.jsonl',
            'CYP2C19_Veith.jsonl',
            'CYP2C9_Veith.jsonl',
            'CYP2D6_Veith.jsonl',
            'CYP3A4_Veith.jsonl',
        ],
        'HTS': [
            'HIV.jsonl',
            'SARSCoV2_3CLPro_Diamond.jsonl',
            'SARSCoV2_Vitro_Touret.jsonl',
            'butkiewicz.jsonl'
        ],
        'Develop': ['SAbDab_Chen.jsonl'],
        'PPI': ['HuRI.jsonl'],
        'TrialOutcome': [
            'phase1.jsonl',
            'phase2.jsonl',
            'phase3.jsonl'
        ],
        'PeptideMHC': [
            'MHC1_IEDB-IMGT_Nielsen.jsonl',
            'MHC2_IEDB_Jensen.jsonl'
        ]
    }

    # Check what actually exists in the directory
    available_files = set(f.name for f in Path(data_dir).glob("*.jsonl"))

    final_map = {}
    for group, filenames in mapping.items():
        existing = [f for f in filenames if f in available_files]
        if existing:
            final_map[group] = existing

    return final_map


def to_prompt_user_block(text: str, tokenizer) -> str:
    """将用户段落经 chat template 展开，并在末尾追加 assistant 起始（生成位点）"""
    conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _extract_gen_logprob(entry, token_id: int):
    """
    从"生成端"的 logprobs（第 0 步）里安全取指定 token 的对数概率。
    vLLM 可能返回 dict[token_id]->Logprob 或对象列表，两种都兼容。
    """
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get(token_id)
        if val is None:
            return None
        lp = getattr(val, "logprob", None)
        if lp is not None:
            return float(lp)
        if isinstance(val, (float, int)):
            return float(val)
        try:
            return float(val)
        except Exception:
            return None
    # 列表形式
    for cand in entry:
        tid = getattr(cand, "token_id", getattr(cand, "id", None))
        if tid == token_id:
            lp = getattr(cand, "logprob", None)
            return float(lp) if lp is not None else None
    return None


def strip_smiles_tags(text: str) -> str:
    return text.replace("<SMILES>", "").replace("</SMILES>", "")


def main():
    args = get_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # ===================== 模型 & tokenizer =====================
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    llm = LLM(
        model=args.model_path,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"video": 0, "image": 0},
        max_logprobs=args.max_logprobs,
        enable_lora=args.use_lora,
    )

    # ★ 关键：只生成 1 个 token，并返回这一"生成步骤"的 top-K 对数概率
    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,  # ★ 只看"下一步"
        logprobs=args.max_logprobs,  # ★ 返回下一步 top-K 候选的 logprobs
        detokenize=False,
    )

    lora_req = None
    if args.use_lora and args.lora_path and os.path.isdir(args.lora_path):
        lora_req = LoRARequest(lora_name='lora',
                               lora_int_id=1,
                               lora_path=args.lora_path)

    # 只比较"首 token"：既考虑 '(A' / '(B'
    A_FIRST_STRS = ["(A"]
    B_FIRST_STRS = ["(B"]

    # 候选"首 token"的 id 集合
    def first_token_ids(cands):
        ids = set()
        for s in cands:
            toks = tokenizer(s, add_special_tokens=False).input_ids
            if len(toks) >= 1:
                ids.add(toks[0])
        return sorted(ids)

    A_FIRST_IDS = first_token_ids(A_FIRST_STRS)
    B_FIRST_IDS = first_token_ids(B_FIRST_STRS)
    logger.info(f"A_FIRST_IDS: {A_FIRST_IDS}, B_FIRST_IDS: {B_FIRST_IDS}")

    # ===================== 数据加载 =====================
    data_path = args.data_dir
    if not data_path.exists():
        logger.error(f"Data directory {data_path} does not exist.")
        return

    task_map = load_tasks_map(data_path)

    groups_to_run = args.task_groups
    if 'all' in groups_to_run:
        groups_to_run = list(task_map.keys())

    logger.info(f"Running groups: {groups_to_run}")

    # 收集所有任务的结果
    all_results = {}

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group}")
            continue

        files = task_map[group]
        all_results[group] = {}

        for filename in files:
            file_path = data_path / filename
            task_name = filename.replace('.jsonl', '')

            logger.info(f"Processing Task: {task_name}")

            # 加载测试数据
            test_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))

            if not test_data:
                logger.warning(f"No data in {filename}")
                continue

            logger.info(f"Total test samples: {len(test_data)}")

            # 准备 prompts
            base_prompts = []
            test_labels = []
            cot_suffix = 'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
            for item in test_data:
                # 数据已经包含完整的 prompt 文本；对齐 legacy 行为，移除 CoT 指令
                raw_text = item['text'].replace(cot_suffix, 'Answer:')
                if args.strip_smiles_tags:
                    raw_text = strip_smiles_tags(raw_text)
                prompt = to_prompt_user_block(raw_text, tokenizer)
                base_prompts.append(prompt)
                test_labels.append(int(item['Y']))

            # ===================== 单批前向：每样本一次 =====================
            logger.info(f"Running inference on {len(base_prompts)} samples...")
            outs = llm.generate(base_prompts, sp, lora_request=lora_req)

            # ===================== 聚合：取 A/B 首 token 的 logprob =====================
            probs, valid_labels, valid_idx = [], [], []

            for i, out in enumerate(outs):
                # 只生成了 1 个 token，因此第 0 步的候选分布在：
                step0 = out.outputs[0].logprobs[0]

                # 在 A/B 的多个首 token 备选中分别取"最大"的一个
                lpA = None
                for tid in A_FIRST_IDS:
                    v = _extract_gen_logprob(step0, tid)
                    if v is not None:
                        lpA = v if lpA is None else max(lpA, v)
                lpB = None
                for tid in B_FIRST_IDS:
                    v = _extract_gen_logprob(step0, tid)
                    if v is not None:
                        lpB = v if lpB is None else max(lpB, v)

                # 若有一边不在 top-K，跳过
                if lpA is None or lpB is None:
                    continue

                pB = float(torch.sigmoid(torch.tensor(lpB - lpA)))  # p = P(B) / (P(A)+P(B))
                if isfinite(pB):
                    probs.append(pB)
                    valid_labels.append(int(test_labels[i]))
                    valid_idx.append(i)

            # ===================== 评估指标计算 =====================
            if len(probs) > 0:
                print("\n" + "="*80)
                print(f"{task_name} EVALUATION RESULTS")
                print("="*80)
                print('Score: p = P("(B)") / ( P("(A)") + P("(B)") ), using next-token logprobs')
                print(f"Valid samples: {len(valid_idx)}/{len(base_prompts)}")

                # 计算AUROC
                if len(set(valid_labels)) > 1:
                    auroc = roc_auc_score(valid_labels, probs)
                    print(f"AUROC: {auroc:.4f}")
                    all_results[group][task_name] = auroc
                else:
                    print("Cannot compute AUROC: only one class in valid samples.")
                    all_results[group][task_name] = None

                print("="*80 + "\n")
            else:
                logger.warning(f"No valid samples for {task_name}")
                all_results[group][task_name] = None

    # ===================== 打印汇总结果 =====================
    print("\n" + "="*80)
    print("SUMMARY RESULTS")
    print("="*80)
    print(f"Model: {args.model_path}")
    print()

    for group in groups_to_run:
        if group not in all_results or not all_results[group]:
            continue
        print(f"\n{group} Tasks:")
        print("-" * 40)

        group_aurocs = []
        for task_name, auroc in all_results[group].items():
            if auroc is not None:
                print(f"  {task_name}: {auroc:.4f}")
                group_aurocs.append(auroc)
            else:
                print(f"  {task_name}: N/A")

        if group_aurocs:
            print(f"  Average: {np.mean(group_aurocs):.4f}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()
