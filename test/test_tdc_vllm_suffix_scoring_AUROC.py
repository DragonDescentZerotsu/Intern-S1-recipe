import os, multiprocessing as mp

# 1) 强制 vLLM 用 spawn，而不是默认的 fork
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # vLLM 官方建议
# 如果你正在使用 vLLM v1（从堆栈看是 vllm.v1），可显式开启 v1 逻辑（多数版本默认已是 v1）
os.environ.setdefault("VLLM_USE_V1", "1")

# 2) 在 Python 侧把多进程 start method 切成 spawn（双保险）
mp.set_start_method("spawn", force=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # TODO: device GPU #

import re
import inspect
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
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Test TDC tasks with vLLM using preprocessed data')

    parser.add_argument('--model-path', type=str,
                        default='Qwen/Qwen3-8B',  # "checkpoints/Intern-S1-mini/full/sft-distill_DeepSeek_V32-TDC-train-set_less_save_interval/checkpoint-3000" # TODO: model name
                        help='Path to the model checkpoint')
                        # internlm/Intern-S1-mini
                        # Qwen/Qwen3-8B
                        # nvidia/NVIDIA-Nemotron-Nano-9B-v2
                        # mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA adapter')
    parser.add_argument('--lora-path', type=str,
                        default="checkpoints/Intern-S1-mini/lora/sft/checkpoint-180000",
                        help='Path to LoRA adapter')
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / "DataPrepare/TDC_test_prompts_label_scaffold",
                        help='Directory containing preprocessed test data')
    parser.add_argument('--task-groups', nargs='+',
                        default=['PPI', 'PeptideMHC'],
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding',
                                 'TrialOutcome', 'PeptideMHC', 'all'],
                        help='Task groups to run')
    parser.add_argument('--max-model-len', type=int, default=1024 * 24, help='Max model length')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92, help='GPU memory utilization')
    parser.add_argument('--max-num-seqs', type=int, default=512, help='Max number of sequences')
    parser.add_argument('--max-logprobs', type=int, default=1024, help='Max logprobs to return')
    # parser.add_argument('--device', type=str, default='2', help='CUDA device ID')
    parser.add_argument('--strip-smiles-tags', action='store_false', help='Remove <SMILES> and </SMILES> from prompts before inference') # TODO: 去掉 <SMILES>
    parser.add_argument('--log-file', action='store_false', help='Enable logging to file')
    parser.add_argument('--log-file-name', type=str, default="test_TDC_single_token_vllm_suffix_scoring_Qwen3-8B_{t_stamp}.log", help='Log file name pattern')  # TODO: log file name

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


def to_prompt_user_block(text: str, tokenizer, model_name: str = None) -> str:
    """
    将用户段落经 chat template 展开，并在末尾追加 assistant 起始（生成位点）
    - 为了兼容更多模型（包括 Qwen3），messages 的 content 用纯字符串更稳
    - enable_thinking 有的 tokenizer 支持、有的不支持；这里做兼容
    """
    messages = []
    kwargs = dict(tokenize=False, add_generation_prompt=True)

    if 'Nemotron' in model_name:
        messages.append({"role": "system", "content": "/no_think"})
        messages.append({"role": "user", "content": text})
    elif "Qwen" in model_name or "Intern" in model_name: # Simple heuristic example, adjust as needed or keep as just passing it through
        kwargs["enable_thinking"] = False
        messages.append({"role": "user", "content": text})
    elif "Mistral" in model_name:
        messages.append({"role": "user", "content": text})
         # kwargs["enable_thinking"] = True # Example of usage
    
    # Original logic was specifically disabling it, let's keep it safe but allow future expansion
   

    return tokenizer.apply_chat_template(messages, **kwargs)


def _extract_gen_logprob(entry, token_id: int):
    """
    从 logprobs entry 里安全取指定 token 的对数概率。
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


def _sum_suffix_logprob(req_out, base_len: int):
    """
    计算 “base_prompt + suffix” 中 suffix 部分 token 的 logprob 总和。
    req_out: vLLM RequestOutput (llm.generate 返回的元素)
    base_len: base_prompt 的 token 数量（不含 suffix）
    """
    plps = getattr(req_out, "prompt_logprobs", None)
    token_ids = getattr(req_out, "prompt_token_ids", None)
    if plps is None or token_ids is None:
        return None
    if base_len >= len(token_ids):
        return None

    total = 0.0
    for pos in range(base_len, len(token_ids)):
        tid = token_ids[pos]
        entry = plps[pos]
        lp = _extract_gen_logprob(entry, tid)
        if lp is None:
            return None
        total += float(lp)
    return total


def strip_smiles_tags(text: str) -> str:
    return text.replace("<SMILES>", "").replace("</SMILES>", "")


def main():
    args = get_args()

    # Set CUDA device
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Configure File Handler if log_file is provided
    if args.log_file:
        current_dir = Path(__file__).parent.resolve()
        log_dir = current_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / args.log_file_name.format(t_stamp=timestamp)
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    # ===================== 模型 & tokenizer =====================
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 一些模型可能没 pad_token（不影响本脚本，但做个兜底）
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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

    lora_req = None
    if args.use_lora and args.lora_path and os.path.isdir(args.lora_path):
        lora_req = LoRARequest(
            lora_name='lora',
            lora_int_id=1,
            lora_path=args.lora_path
        )

    # ===================== suffix scoring 的 SamplingParams =====================
    # 关键：prompt_logprobs 返回 prompt 每个 token 的 logprob
    # max_tokens=1 主要是兼容性更好（我们不会用生成 token）
    sp_score = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )

    # suffix 候选：尽量覆盖常见格式（空格/换行）
    # 注意：这里的 suffix 是加在 assistant 生成位点之后，不会和 "Answer:" 发生 token merge 的坑
    A_SUFFIXES = ["(A"]#, " (A)", "\n(A)", "\n (A)"]
    B_SUFFIXES = ["(B"]#, " (B)", "\n(B)", "\n (B)"]B
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

            # ===================== 准备 prompts =====================
            base_prompts = []
            test_labels = []
            cot_suffix = 'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'

            for item in test_data:
                # 数据已经包含完整的 prompt 文本；对齐 legacy 行为，移除 CoT 指令
                raw_text = item['text'].replace(cot_suffix, 'Answer:')

                # （可选）为了更稳定的边界，也可以强制 Answer: 后跟一个空格
                # raw_text = re.sub(r'Answer:\s*', 'Answer: ', raw_text)

                if args.strip_smiles_tags:
                    raw_text = strip_smiles_tags(raw_text)

                prompt = to_prompt_user_block(raw_text, tokenizer, model_name=args.model_path)
                base_prompts.append(prompt)
                test_labels.append(int(item['Y']))

            # base_prompt 的 token 长度（用于切 suffix 段）
            base_lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in base_prompts]

            # ===================== suffix scoring 推理 =====================
            # 我们对每个样本构造多个 (A)/(B) 候选，取最大得分作为该类得分
            bestA = [None] * len(base_prompts)
            bestB = [None] * len(base_prompts)

            # 构造 batch candidates
            batch_prompts = []
            meta = []  # (sample_idx, 'A' or 'B')

            for i, p in enumerate(base_prompts):
                for suf in A_SUFFIXES:
                    batch_prompts.append(p + suf)
                    meta.append((i, "A"))
                for suf in B_SUFFIXES:
                    batch_prompts.append(p + suf)
                    meta.append((i, "B"))

            logger.info(
                f"Sanity check: prompt+suffix candidate:\n  {batch_prompts[0]}\n"
                f"Suffix-scoring: {len(base_prompts)} samples -> {len(batch_prompts)} candidates "
                f"(A:{len(A_SUFFIXES)} + B:{len(B_SUFFIXES)} each)"
            )

            # 一次性执行所有 candidates
            logger.info(f"Running inference on {len(batch_prompts)} candidates...")
            outs = llm.generate(batch_prompts, sp_score, lora_request=lora_req)

            for out, (idx, lab) in zip(outs, meta):
                lp = _sum_suffix_logprob(out, base_lens[idx])
                if lp is None:
                    continue
                if lab == "A":
                    bestA[idx] = lp if bestA[idx] is None else max(bestA[idx], lp)
                else:
                    bestB[idx] = lp if bestB[idx] is None else max(bestB[idx], lp)

            # ===================== 聚合：p(B)=sigmoid(scoreB-scoreA) =====================
            probs, valid_labels, valid_idx = [], [], []

            for i in range(len(base_prompts)):
                lpA, lpB = bestA[i], bestB[i]
                if lpA is None or lpB is None:
                    continue
                pB = float(torch.sigmoid(torch.tensor(lpB - lpA)))
                if isfinite(pB):
                    probs.append(pB)
                    valid_labels.append(int(test_labels[i]))
                    valid_idx.append(i)

            # ===================== 评估指标计算 =====================
            if len(probs) > 0:
                logger.info("\n" + "=" * 80)
                logger.info(f"{task_name} EVALUATION RESULTS")
                logger.info("=" * 80)
                logger.info('Score: p = sigmoid( score_B - score_A ), where score is SUM of suffix token logprobs')
                logger.info(f"Valid samples: {len(valid_idx)}/{len(base_prompts)}")

                # 计算 AUROC
                if len(set(valid_labels)) > 1:
                    auroc = roc_auc_score(valid_labels, probs)
                    logger.info(f"AUROC: {auroc:.4f}")
                    all_results[group][task_name] = auroc
                else:
                    logger.warning("Cannot compute AUROC: only one class in valid samples.")
                    all_results[group][task_name] = None

                logger.info("=" * 80 + "\n")
            else:
                logger.warning(f"No valid samples for {task_name}")
                all_results[group][task_name] = None

    # ===================== 打印汇总结果 =====================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info("")

    for group in groups_to_run:
        if group not in all_results or not all_results[group]:
            continue
        logger.info(f"\n{group} Tasks:")
        logger.info("-" * 40)

        group_aurocs = []
        for task_name, auroc in all_results[group].items():
            if auroc is not None:
                logger.info(f"  {task_name}: {auroc:.4f}")
                group_aurocs.append(auroc)
            else:
                logger.info(f"  {task_name}: N/A")

        if group_aurocs:
            logger.info(f"  Average: {np.mean(group_aurocs):.4f}")

    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
