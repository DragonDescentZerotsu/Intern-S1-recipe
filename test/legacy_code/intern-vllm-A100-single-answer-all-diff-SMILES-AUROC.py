import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from math import isfinite
from rdkit import Chem
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

# ===================== 配置 =====================
MODEL_NAME = "internlm/Intern-S1-mini"
TASK_NAME = "CYP2C9_Substrate_CarbonMangels"
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
    # quantization="fp8",
    dtype="bfloat16",
    tensor_parallel_size=8 if MODEL_NAME == "internlm/Intern-S1" else 1,
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
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

# ===================== SMILES Robustness ==========================
print("\n" + "="*80)
print("ROBUSTNESS EVALUATION ACROSS SMILES VARIANTS (rooted SMILES via RDKit)")
print("="*80)

# 1) 用 RDKit 为每个 SMILES 生成“rooted SMILES”变体：把每个原子作为根节点（去重）
def enumerate_rooted_smiles_variants(smi: str, include_original: bool = True):
    """
    对给定 SMILES，返回一个去重后的 rooted SMILES 列表。
    rooted 的含义：MolToSmiles(..., rootedAtAtom=i, canonical=False, isomericSmiles=True)
    这样能得到以不同原子为起点/根的不同写法，从而覆盖“同一个分子在不同 SMILES 表达下”的变体。
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    n = mol.GetNumAtoms()
    variants = set()

    # 可选：保留原始写法
    if include_original:
        try:
            orig = Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True)
            if orig:
                variants.add(orig)
        except Exception:
            pass

    # 枚举每个原子做根节点
    for i in range(n):
        try:
            rs = Chem.MolToSmiles(
                mol,
                rootedAtAtom=i,
                canonical=False,          # 必须设为 False 才会保留 rooted 顺序
                isomericSmiles=True       # 保留立体信息
            )
            if rs:
                variants.add(rs)
        except Exception:
            # 个别分子可能在某些根节点失败，忽略即可
            continue

    return sorted(variants)

# 2) 为所有 test_drugs 生成各自的变体，并构建 prompts
variant_prompts = []
variant_parent_idx = []   # 记录此变体属于哪个原始样本 i
variant_smiles_list = []  # 保存具体变体的 SMILES

print("Enumerating rooted SMILES for each test sample ...")
pbar = tqdm(total=len(test_drugs), desc="Enumerating rooted SMILES")
for i, smi in enumerate(testi := list(test_drugs)):  # test_drugs 在上文已有
    vs = enumerate_rooted_smiles_variants(smi, include_original=True)
    if not vs:
        continue
    for v_smi in vs:
        user_text = tdc_prompts_json[TASK_NAME].replace(
            INPUT_TYPE, f'<SMILES>{v_smi}</SMILES>'
        )
        variant_prompts.append(to_prompt_user_block(user_text))
        variant_parent_idx.append(i)
        variant_smiles_list.append(v_smi)
    pbar.update(1)
pbar.close()

print(f"Total variant prompts: {len(variant_prompts)} "
      f"(avg {len(variant_prompts)/max(1,len(testi)):.2f} per sample)")

# 3) 统一前向：读取“下一 token”的 logprobs（与主流程一致）
#    如变体过多，可分批以节约显存
def batched(iterable, batch_size):
    for start in range(0, len(iterable), batch_size):
        yield start, iterable[start:start + batch_size]

print("Scoring all variants (single-step generation, next-token logprobs) ...")
outs_v = llm.generate(variant_prompts, sp)
variant_out_logprobs = [None] * len(variant_prompts)
for i, out in enumerate(outs_v):
    # 只生成了 1 个 token：第 0 步的候选分布
    step0 = out.outputs[0].logprobs[0] if out and out.outputs else None
    variant_out_logprobs[i]=step0

# 4) 计算每个变体的 pB，然后聚合到各自的原始样本，得到该样本下 pB 的 min/max（以及范围）
from collections import defaultdict

per_parent_pBs = defaultdict(list)   # parent_idx -> [pB_of_each_variant]
per_parent_seen = defaultdict(list)  # 记录对应的变体 SMILES（可调试用）

for k, step0 in enumerate(variant_out_logprobs):
    if step0 is None:
        continue

    # 从 A/B 候选首 token 中取“最大”logprob（兼容 '(A'/'A' 等）
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

    if lpA is None or lpB is None:
        continue

    pB = float(torch.sigmoid(torch.tensor(lpB - lpA)))  # P(B)/(P(A)+P(B))
    if not isfinite(pB):
        continue

    p_idx = variant_parent_idx[k]
    per_parent_pBs[p_idx].append(float(pB))
    per_parent_seen[p_idx].append(variant_smiles_list[k])

# 5) 计算每个样本的误差范围（max - min），并画直方图；同时准备“最优/最差 pB”用于 AUROC
import numpy as np
import matplotlib.pyplot as plt

ranges = []
best_probs = []   # 相对 ground truth 的“最有利 pB”
worst_probs = []  # 相对 ground truth 的“最不利 pB”
used_labels = []  # 对应的 label 列表（与 best/worst 对齐）
used_idx = []     # 用于统计有效样本数

print("\nAggregating per-sample pB stats across variants ...")
for i, label in enumerate(test_labels):
    pBs = per_parent_pBs.get(i, [])
    if len(pBs) == 0:
        continue
    p_min = float(np.min(pBs))
    p_max = float(np.max(pBs))
    ranges.append(p_max - p_min)

    # “根据 ground truth 来看”的最优/最差：
    # 原脚本已以 pB 对应 label=1 来算 AUROC，因此延续该语义：
    # 若 y=1：pB 越大越“有利”（best 取 max，worst 取 min）
    # 若 y=0：pB 越小越“有利”（best 取 min，worst 取 max）
    if int(label) == 1:
        best_probs.append(p_max)
        worst_probs.append(p_min)
    else:
        best_probs.append(p_min)
        worst_probs.append(p_max)
    used_labels.append(int(label))
    used_idx.append(i)

ranges = np.array(ranges, dtype=float)

print(f"Valid parent samples with >=1 variant scored: {len(used_idx)}/{len(test_drugs)}")
if len(ranges) > 0:
    print(f"Range stats (max-min of pB across variants): "
          f"mean={ranges.mean():.4f}, std={ranges.std():.4f}, "
          f"median={np.median(ranges):.4f}, max={ranges.max():.4f}")

    # 画一个直方图（统计图）
    figs_dir = current_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    plt.hist(ranges, bins=40)
    plt.xlabel("pB range across rooted SMILES variants (max - min)")
    plt.ylabel("Count")
    plt.title("Per-sample pB variability across SMILES permutations")
    plt.tight_layout()

    fig_path = figs_dir / f"{MODEL_NAME.split('/')[-1]}_{TASK_NAME}_pB_variability.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()
else:
    print("No per-sample ranges available (no variants scored).")

# 6) 基于“最优 / 最差 pB”计算两版 AUROC
best_auroc, worst_auroc = None, None
if len(set(used_labels)) > 1 and len(best_probs) > 0:
    best_auroc = roc_auc_score(used_labels, best_probs)
    worst_auroc = roc_auc_score(used_labels, worst_probs)

    print("\n" + "="*80)
    print("AUROC UNDER VARIANT SELECTION (w.r.t. ground truth)")
    print("="*80)
    print(f"Best-case AUROC : {best_auroc:.4f}  "
          f"(choose pB=max if y=1 else min)")
    print(f"Worst-case AUROC: {worst_auroc:.4f}  "
          f"(choose pB=min if y=1 else max)")
    print("="*80 + "\n")
else:
    print("Cannot compute best/worst AUROC: not enough valid samples or only one class.")

# （可选）打印前 3 个样本的详细变体情况，看看差异
for _j, i in enumerate(used_idx[:3]):
    pBs = per_parent_pBs.get(i, [])
    if not pBs:
        continue
    print(f"[Parent sample {i}] y={int(test_labels[i])}, "
          f"variants={len(pBs)}, min={min(pBs):.4f}, max={max(pBs):.4f}")