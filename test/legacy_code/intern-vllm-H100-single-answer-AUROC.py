import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from math import isfinite

# ===================== 配置 =====================
MODEL_NAME = "internlm/Intern-S1-mini-FP8"
TASK_GROUP_NAME = 'Tox' # 'ADME'
TASK_NAMEs = ["hERG", "Carcinogens_Lagunin"]
INPUT_TYPE = "{Drug SMILES}"

# 只比较“首 token”：既考虑 '(A' / '(B'，也可选兼容 'A' / 'B'
A_FIRST_STRS = ["(A"]#, "A"]   # ← 如只想用 '(A' 就把 "A" 删掉
B_FIRST_STRS = ["(B"]#, "B"]

# ===================== 模型 & tokenizer =====================
# processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    gpu_memory_utilization=0.9,
    max_num_seqs=128,
    limit_mm_per_prompt={"video": 0, "image": 0},
    max_logprobs=1024,  # ↑ 提高返回 top-K 的上限，提高命中率
)

# ★ 关键：只生成 1 个 token，并返回这一“生成步骤”的 top-K 对数概率
sp = SamplingParams(
    temperature=0.0,    # 只影响采样选择，不影响我们读取到的分布值
    top_p=1.0,          # 防止采样截断影响候选覆盖（与返回的logprobs无冲突）
    max_tokens=1,       # ★ 只看“下一步”
    logprobs=1024,      # ★ 返回下一步 top-K 候选的 logprobs（设大些）
    detokenize=False,
)

# ===================== 实用函数 =====================
def to_prompt_user_block(text: str) -> str:
    """将用户段落经 chat template 展开，并在末尾追加 assistant 起始（生成位点）"""
    conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def _extract_gen_logprob(entry, token_id: int):
    """
    从“生成端”的 logprobs（第 0 步）里安全取指定 token 的对数概率。
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

# ===================== 数据与 prompt =====================
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-predict",
    filename="tdc_prompts.json",
)
tdc_prompts_json = json.load(open(tdc_prompts_filepath, "r"))

for TASK_NAME in TASK_NAMEs:
    if TASK_GROUP_NAME == 'Tox':
        data = Tox(name=TASK_NAME)
    if TASK_GROUP_NAME == 'ADME':
        data = ADME(name=TASK_NAME)
    split = data.get_split()
    test_drugs  = split['test']['Drug'].values
    test_labels = split['test']['Y'].values  # 0/1

    print(f"Total test samples: {len(test_drugs)}")

    # 评分用的 prompt：保留原模板里的 Answer:（不要注入“先思考再回答”）
    base_prompts = []
    for smi in test_drugs:
        user_text = tdc_prompts_json[TASK_NAME].replace(
            INPUT_TYPE, f'<SMILES>{smi}</SMILES>'
        )
        base_prompts.append(to_prompt_user_block(user_text))

    # 候选“首 token”的 id 集合（不同分词表下 '(A' 可能不是单 token，用第一枚即可）
    def first_token_ids(cands):
        ids = set()
        for s in cands:
            toks = tokenizer(s, add_special_tokens=False).input_ids
            if len(toks) >= 1:
                ids.add(toks[0])
        return sorted(ids)

    A_FIRST_IDS = first_token_ids(A_FIRST_STRS)
    B_FIRST_IDS = first_token_ids(B_FIRST_STRS)
    print("A_FIRST_IDS:", A_FIRST_IDS, "B_FIRST_IDS:", B_FIRST_IDS)

    # ===================== 单批前向：每样本一次 =====================
    print("Running single-step generation (one forward per sample) to read next-token logprobs ...")
    outs = llm.generate(base_prompts, sp)

    # ===================== 聚合：取 A/B 首 token 的 logprob（在第 0 个生成步的候选里找） =====================
    probs, valid_labels, valid_idx = [], [], []

    for i, out in enumerate(outs):
        # 只生成了 1 个 token，因此第 0 步的候选分布在：
        step0 = out.outputs[0].logprobs[0]

        # 在 A/B 的多个首 token 备选中分别取“最大”的一个（避免切分差异丢失）
        lpA = None
        for tid in A_FIRST_IDS:
            v = _extract_gen_logprob(step0, tid)
            if v is not None:
                lpA = v if lpA is None else max(lpA, v)  # 还是会
        lpB = None
        for tid in B_FIRST_IDS:
            v = _extract_gen_logprob(step0, tid)
            if v is not None:
                lpB = v if lpB is None else max(lpB, v)

        # 若有一边不在 top-K，增大 sp.logprobs 或引擎 max_logprobs 再试
        if lpA is None or lpB is None:
            continue

        pB = float(torch.sigmoid(torch.tensor(lpB - lpA)))  # p = P(B) / (P(A)+P(B))
        if isfinite(pB):
            probs.append(pB)
            valid_labels.append(int(test_labels[i]))
            valid_idx.append(i)

    # ===================== AUROC =====================
    if len(set(valid_labels)) > 1 and len(probs) > 0:
        auroc = roc_auc_score(valid_labels, probs)
        print("\n" + "="*80)
        print("EVALUATION RESULTS (single-forward, next-token logprobs)")
        print("="*80)
        print('Score: p = P("(B)") / ( P("(A)") + P("(B)") ), using next-token logprobs (generation step)')
        print(f"Valid samples: {len(valid_idx)}/{len(base_prompts)}")
        print(f"AUROC: {auroc:.4f}")
        print("="*80 + "\n")
    else:
        print("Cannot compute AUROC: not enough valid samples or only one class.")

    # （可选）调试前 3 个样本
    for j in range(min(3, len(valid_idx))):
        i = valid_idx[j]
        print(f"[Sample {i}] p={probs[j]:.4f}, y={valid_labels[j]}")
