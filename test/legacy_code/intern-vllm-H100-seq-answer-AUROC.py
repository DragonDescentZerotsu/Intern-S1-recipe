import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from math import isfinite

# ===================== 配置 =====================
MODEL_NAME = "internlm/Intern-S1-mini-FP8"
TASK_NAME = "BBB_Martins"
INPUT_TYPE = "{Drug SMILES}"

# 兼容两种“首 token”划分：(A vs A；也顺带给 B 做对称处理)
A_VARIANTS = ["(A"]#, "A", " (A)", " A", "\n(A)", "\nA"]
B_VARIANTS = ["(B"]#, "B", " (B)", " B", "\n(B)", "\nB"]

# 如果一次性全量会 OOM，可把它设成较大的分块，例如 2048/4096
BATCH_CHUNK = None  # None = 单批；否则按这个大小切分

# ===================== 模型 & tokenizer =====================
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

llm = LLM(
    model=MODEL_NAME,
    enforce_eager=True,
    max_model_len=1024 * 8,
    max_num_batched_tokens=1024 * 8,
    quantization="fp8",
    dtype="bfloat16",
    tensor_parallel_size=4 if MODEL_NAME == "internlm/Intern-S1-FP8" else 1,
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
    max_num_seqs=128,
    limit_mm_per_prompt={"video": 0, "image": 0},
    max_logprobs=512,  # ★ 允许较大的 top-k，提升真实 token 覆盖率
)

sp = SamplingParams(
    temperature=0.0,
    max_tokens=1,          # ★ 只做 prefill，不生成
    prompt_logprobs=512,   # ★ 返回 prompt 端每个位置的 top-k logprobs
    detokenize=False,
)

# ===================== 实用函数 =====================
def to_prompt_user_block(text: str) -> str:
    """将用户段落经 chat template 展开，并在末尾追加 assistant 起始"""
    conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def _extract_logprob(entry, token_id: int):
    """兼容 vLLM 不同返回结构，安全取出指定 token_id 的对数概率"""
    if entry is None:
        return None
    # ① 常见：字典 {token_id -> Logprob 或 float}
    if isinstance(entry, dict):
        val = entry.get(token_id)
        if val is None:
            return None
        # vLLM 通常返回 Logprob 对象，取其 .logprob
        lp = getattr(val, "logprob", None)
        if lp is not None:
            return float(lp)
        # 少数旧版本/兼容路径可能直接是 float
        if isinstance(val, (float, int)):
            return float(val)
        # 再兜底一次（防止别的包装类型）
        try:
            return float(val)
        except Exception:
            return None
    # ② 另一种：列表，每个元素带 .token_id / .logprob
    for cand in entry:
        cand_id = getattr(cand, "token_id", getattr(cand, "id", None))
        if cand_id == token_id:
            lp = getattr(cand, "logprob", None)
            if lp is not None:
                return float(lp)
            try:
                return float(cand)
            except Exception:
                return None
    return None

def sum_tail_logprob(out, option_ids):
    """对 out 的末尾 |option_ids| 个 prompt 位置，取真实 token 的 logprob 并相加。
       若某位置真实 token 不在 top-k，返回 None。"""
    k = len(option_ids)
    all_ids = out.prompt_token_ids
    start = len(all_ids) - k
    total = 0.0
    for pos, tid in zip(range(start, len(all_ids)), option_ids):
        lp = _extract_logprob(out.prompt_logprobs[pos], tid)
        if lp is None:
            return None
        total += lp
    return total

# ===================== 数据与 prompt =====================
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-predict",
    filename="tdc_prompts.json",
)
tdc_prompts_json = json.load(open(tdc_prompts_filepath, "r"))

data = ADME(name=TASK_NAME)
split = data.get_split()
test_drugs  = split['test']['Drug'].values
test_labels = split['test']['Y'].values  # 0/1

print(f"Total test samples: {len(test_drugs)}")

# scoring 用的 prompt：保留原模板里的 Answer:（不要注入“先思考再回答”）
base_prompts = []
for smi in test_drugs:
    user_text = tdc_prompts_json[TASK_NAME].replace(
        INPUT_TYPE, f'<SMILES>{smi}</SMILES>'
    )
    base_prompts.append(to_prompt_user_block(user_text))

# 预分词：每个变体 -> token id 序列
A_VAR_IDS = {v: tokenizer(v, add_special_tokens=False).input_ids for v in A_VARIANTS}
B_VAR_IDS = {v: tokenizer(v, add_special_tokens=False).input_ids for v in B_VARIANTS}

# ===================== 组装“单批请求” =====================
# 我们把所有样本 × (A 的各变体 + B 的各变体) 拼成一个大列表，一次性丢给 vLLM
requests = []   # [(sample_idx, 'A'/'B', variant_str, option_ids, full_prompt_str)]
for i, p in enumerate(base_prompts):
    for v, ids in A_VAR_IDS.items():
        requests.append((i, 'A', v, ids, p + v))
    for v, ids in B_VAR_IDS.items():
        requests.append((i, 'B', v, ids, p + v))

def run_generate_in_chunks(reqs):
    outs = []
    if BATCH_CHUNK is None:
        texts = [r[4] for r in reqs]
        outs = llm.generate(texts, sp)
    else:
        for s in range(0, len(reqs), BATCH_CHUNK):
            texts = [r[4] for r in reqs[s:s+BATCH_CHUNK]]
            outs.extend(llm.generate(texts, sp))
    return outs

print(f"Submitting {len(requests)} requests to vLLM in "
      f"{'one batch' if BATCH_CHUNK is None else f'chunks of {BATCH_CHUNK}'} ...")

outputs = run_generate_in_chunks(requests)

# ===================== 聚合：每个样本取 A/B 变体中的最大 logprob =====================
n = len(base_prompts)
best_lp_A = [None] * n
best_lp_B = [None] * n

for meta, out in zip(requests, outputs):
    i, which, v, opt_ids, _ = meta
    lp = sum_tail_logprob(out, opt_ids)
    if lp is None:
        continue
    if which == 'A':
        best_lp_A[i] = lp if best_lp_A[i] is None else max(best_lp_A[i], lp)
    else:
        best_lp_B[i] = lp if best_lp_B[i] is None else max(best_lp_B[i], lp)

# 取有效样本：A 与 B 都拿到了 logprob
valid_indices, probs = [], []
for i in range(n):
    if best_lp_A[i] is None or best_lp_B[i] is None:
        continue
    pB = float(torch.sigmoid(torch.tensor(best_lp_B[i] - best_lp_A[i])))
    if isfinite(pB):
        valid_indices.append(i)
        probs.append(pB)

valid_labels = [int(test_labels[i]) for i in valid_indices]

# 调试输出前 3 个样本
for j in range(min(3, len(valid_indices))):
    i = valid_indices[j]
    print(f"\n[Sample {i}] lp(A)={best_lp_A[i]:.4f}, lp(B)={best_lp_B[i]:.4f}, "
          f"p={probs[j]:.4f}, y={valid_labels[j]}")

# ===================== AUROC =====================
if len(set(valid_labels)) > 1 and len(probs) > 0:
    auroc = roc_auc_score(valid_labels, probs)
    print("\n" + "="*80)
    print("EVALUATION RESULTS (batched probability-based via prompt logprobs)")
    print("="*80)
    print('Score: p = P("(B)") / ( P("(A)") + P("(B)") ), using prompt logprobs (prefill)')
    print(f"Valid samples: {len(valid_indices)}/{n}")
    print(f"AUROC: {auroc:.4f}")
    print("="*80 + "\n")
else:
    print("Cannot compute AUROC: not enough valid samples or only one class.")
