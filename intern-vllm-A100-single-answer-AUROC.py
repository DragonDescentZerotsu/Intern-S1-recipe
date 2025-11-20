import os, multiprocessing as mp

# 1) 强制 vLLM 用 spawn，而不是默认的 fork
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # vLLM 官方建议
# 如果你正在使用 vLLM v1（从堆栈看是 vllm.v1），可显式开启 v1 逻辑（多数版本默认已是 v1）
os.environ.setdefault("VLLM_USE_V1", "1")

# 2)（可选）暂时禁用 flash-attn 的后端检查，绕开导入阶段对 CUDA 的访问
# 如需完全规避，可切到 PyTorch SDPA 注意力：
#  - 好处：不会在导入时访问 CUDA
#  - 代价：性能可能略降
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")

# 3) 在 Python 侧把多进程 start method 切成 spawn（双保险）
mp.set_start_method("spawn", force=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI  #, TCREpitopeBinding, TrialOutcome, PeptideMHC
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from math import isfinite
import numpy as np
import ast

def main():
    # ===================== 配置 =====================
    MODEL_NAME = "internlm/Intern-S1-mini"
    # MODEL_NAME = "checkpoints/Intern-S1-mini/full/sft"
    USE_LORA = False
    LORA_PATH = "checkpoints/Intern-S1-mini/lora/sft/checkpoint-180000"  # 你的 LoRA 目录
    # ===================== 模型 & tokenizer =====================
    # processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=1024 * 24,
        max_num_batched_tokens=1024 * 24,
        # quantization="fp8",
        dtype="bfloat16",
        tensor_parallel_size=1 if 'Intern-S1-mini' in MODEL_NAME else 8,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_num_seqs=256,
        limit_mm_per_prompt={"video": 0, "image": 0},
        max_logprobs=1024,  # ↑ 提高返回 top-K 的上限，提高命中率
        enable_lora=USE_LORA,
    )

    # ★ 关键：只生成 1 个 token，并返回这一“生成步骤”的 top-K 对数概率
    sp = SamplingParams(
        temperature=0.0,    # 只影响采样选择，不影响我们读取到的分布值
        top_p=1.0,          # 防止采样截断影响候选覆盖（与返回的logprobs无冲突）
        max_tokens=1,       # ★ 只看“下一步”
        logprobs=1024,      # ★ 返回下一步 top-K 候选的 logprobs（设大些）
        detokenize=False,
    )

    lora_req = None
    if USE_LORA and LORA_PATH and os.path.isdir(LORA_PATH):
        lora_req = LoRARequest(lora_name='lora',
                               lora_int_id=1,
                               lora_path=LORA_PATH) # 你的 LoRA 目录

    TASK_GROUP_NAMEs = ['ADME', 'Tox'] # 'ADME'/'Tox'/'HTS'/'Develop'/'PPI'/'TCREpitopeBinding'/'TrialOutcome'/'PeptideMHC'

    AUTO_FASTA = True
    if not AUTO_FASTA:
        FASTA_WRAP_TOKEN_PAIR = {'start':'<FASTA>',
                                 'end':'</FASTA>'}
    else:
        FASTA_WRAP_TOKEN_PAIR = {'start':'',
                                 'end':''}
    # Tox
    ToxCast_AUROCS = []
    herg_central_AUROCS = []
    Tox21_AUROCS = []
    Skin_Reaction_AUROCS = []
    hERG_AUROCS = []
    AMES_AUROCS = []
    DILI_AUROCS = []
    ClinTox_AUROCS = []

    # ADME
    PAMPA_NCATS_AUROCS = []
    HIA_Hou_AUROCS = []
    Bioavailability_Ma_AUROCS = []
    BBB_Martins_AUROCS = []
    Pgp_Broccatelli_AUROCS = []
    CYP1A2_Veith_AUROCS = []
    CYP2C19_Veith_AUROCS = []
    CYP2C9_Veith_AUROCS = []
    CYP2D6_Veith_AUROCS = []
    CYP3A4_Veith_AUROCS = []
    CYP2C9_Substrate_CarbonMangels_AUROCS = []
    CYP2D6_Substrate_CarbonMangels_AUROCS = []
    CYP3A4_Substrate_CarbonMangels_AUROCS = []

    # HTS
    HIV_AUROCS = []
    SARSCoV2_3CLPro_Diamond_AUROCS = []
    SARSCoV2_Vitro_Touret_AUROCS = []
    butkiewicz_AUROCS = []

    # Develop
    SAbDab_Chen_AUROCS = []

    # PPI
    HuRI_AUROCS = []

    # TCREpitopeBinding
    Weber_AUROCS = []

    # TrialOutcome

    # PeptideMHC
    MHC1_IEDB_IMGT_Nielsen_AUROCS = []
    MHC2_IEDB_Jensen_AUROCS = []


    INPUT_TYPE = "{Drug SMILES}"
    METRIC_TYPE = "auroc"  # 可选："auroc", "accuracy", "both"
    ACCURACY_THRESHOLD = 0.5  # accuracy分类阈值

    # 只比较“首 token”：既考虑 '(A' / '(B'，也可选兼容 'A' / 'B'
    A_FIRST_STRS = ["(A"]#, "A"]   # ← 如只想用 '(A' 就把 "A" 删掉
    B_FIRST_STRS = ["(B"]#, "B"]

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
    # fix a key bug
    tdc_prompts_json['SARSCoV2_3CLPro_Diamond'] = tdc_prompts_json['SARSCOV2_3CLPro_Diamond']

    for TASK_GROUP_NAME in TASK_GROUP_NAMEs:
        # single-pred
        if TASK_GROUP_NAME == 'Tox':
            # Accuracy
            # TASK_NAMEs = ["hERG_Karim", "Carcinogens_Lagunin"]
            # AUROC
            TASK_NAMEs = (
                          ['Skin_Reaction', 'hERG', 'AMES', 'DILI', 'ClinTox'] +
                          ['Tox21'+'_'+label.replace('-', '_') for label in retrieve_label_name_list('Tox21')] +
                          ['herg_central'+'_'+retrieve_label_name_list('herg_central')[-1]] +
                          ['ToxCast'+'_'+label for label in retrieve_label_name_list('Toxcast')]
                          )
        elif TASK_GROUP_NAME == 'ADME':
            TASK_NAMEs = [
                # Absorption / Distribution
                'PAMPA_NCATS',       # Binary classification
                'HIA_Hou',           # Binary classification
                'Bioavailability_Ma',# Binary classification
                'BBB_Martins',       # Binary classification
                'Pgp_Broccatelli',   # Binary classification

                # Metabolism (CYP inhibition / substrate) — all binary
                'CYP1A2_Veith',
                'CYP2C19_Veith',
                'CYP2C9_Veith',
                'CYP2D6_Veith',
                'CYP3A4_Veith',
                'CYP2C9_Substrate_CarbonMangels',
                'CYP2D6_Substrate_CarbonMangels',
                'CYP3A4_Substrate_CarbonMangels',
            ]
        elif TASK_GROUP_NAME == 'HTS':
            TASK_NAMEs = [
                'cav3_t-type_calcium_channels_butkiewicz',
                'tyrosyl-dna_phosphodiesterase_butkiewicz',
                'HIV',
                'SARSCoV2_3CLPro_Diamond',
                'SARSCoV2_Vitro_Touret',
                # Butkiewicz 9 个二分类数据集
                'orexin1_receptor_butkiewicz',
                'm1_muscarinic_receptor_agonists_butkiewicz',
                'm1_muscarinic_receptor_antagonists_butkiewicz',
                'potassium_ion_channel_kir2.1_butkiewicz',
                'kcnq2_potassium_channel_butkiewicz',
                'choline_transporter_butkiewicz',
                'serine_threonine_kinase_33_butkiewicz',
            ]
        elif TASK_GROUP_NAME == 'Develop':
            TASK_NAMEs = [
                'SAbDab_Chen',   # Binary classification
                # 另一个 TAP 是回归，不要放进你当前的二分类脚本
            ]

        # multi-pred
        # 例：TASK_GROUP_NAME = 'PPI' / 'TCREpitopeBinding' / 'TrialOutcome' / 'PeptideMHC'
        elif TASK_GROUP_NAME == 'PPI':
            TASK_NAMEs = ['HuRI']  # 目前 multipred 的 PPI 常用就是 HuRI
        elif TASK_GROUP_NAME == 'TCREpitopeBinding':
            TASK_NAMEs = ['Weber']  # TCR-epitope（Weber 数据集）
        elif TASK_GROUP_NAME == 'TrialOutcome':  # TODO: no TxGemma template for this task yet?
            TASK_NAMEs = [
                'phase1',
                'phase2',
                'phase3'
            ]  # 临床试验 1/2/3 期
        elif TASK_GROUP_NAME == 'PeptideMHC':
            # 可选：MHC-I 或 MHC-II；不同子集在 TDC 里通过 name 指定
            TASK_NAMEs = ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']  # 你也可以换成 'IEDB_MHC_II' 等

        for TASK_NAME in tqdm(TASK_NAMEs, desc="Processing tasks ..."):
            if TASK_GROUP_NAME == 'Tox':
                data = None
                for special_task in ['ToxCast', 'herg_central', 'Tox21']:
                    if TASK_NAME.startswith(special_task):
                        data = Tox(name=special_task,
                                   label_name = TASK_NAME.split(special_task+'_')[-1] if special_task!='Tox21' else TASK_NAME.split(special_task+'_')[-1].replace('_', '-'))
                        break
                if data is None:
                    data = Tox(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'ADME':
                data = ADME(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'HTS':
                data = HTS(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'Develop':
                data = Develop(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'PPI':
                data = PPI(name=TASK_NAME)
                data = data.neg_sample(frac=1)  # special for PPI dataset (HuRI), otherwise there is no negative samples
            elif TASK_GROUP_NAME == 'TCREpitopeBinding':
                data = TCREpitopeBinding(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'TrialOutcome':
                data = TrialOutcome(name=TASK_NAME)
            elif TASK_GROUP_NAME == 'PeptideMHC':
                data = PeptideMHC(name=TASK_NAME)
            split = data.get_split()
            if TASK_NAME == 'SAbDab_Chen':
                test_drugs  = split['test']['Antibody'].values
            elif TASK_NAME == 'HuRI':
                test_drugs = split['test'][['Protein1', 'Protein2']].values
            elif TASK_NAME == 'Weber':
                # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
                test_drugs = split['test'][['epitope_aa', 'tcr']].values
            elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
                test_drugs = split['test'][['Peptide', 'MHC']].values
            else:
                test_drugs  = split['test']['Drug'].values

            if TASK_NAME in ['Weber']:
                test_labels = split['test']['label'].values
            else:
                test_labels = split['test']['Y'].values  # 0/1

            print(f"Total test samples: {len(test_drugs)}")

            # 评分用的 prompt：保留原模板里的 Answer:（不要注入“先思考再回答”）
            base_prompts = []
            for smi in test_drugs:
                if TASK_NAME.startswith('herg_central'):
                    TASK_NAME = 'herg_central'
                if TASK_NAME == 'SAbDab_Chen':
                    smi = ast.literal_eval(smi)
                    user_text = tdc_prompts_json[TASK_NAME.replace('-', '_')].replace(
                        '{Antibody heavy chain sequence}', f'<FASTA>{smi[0]}</FASTA>'
                    ).replace(
                        '{Antibody light chain sequence}', f'<FASTA>{smi[1]}</FASTA>'
                    )
                    # print(user_text)
                    # exit(1)
                elif TASK_NAME == 'HuRI':
                    user_text = tdc_prompts_json[TASK_NAME.replace('-', '_')].replace(
                        '{Protein1 amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                    ).replace(
                        '{Protein2 amino acid sequence}', f'<FASTA>{smi[1]}</FASTA>'
                    )
                    # print(user_text)
                    # exit(1)
                elif TASK_NAME == 'Weber':
                    user_text = tdc_prompts_json['weber'].replace(
                        '{Epitope amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                    ).replace(
                        '{TCR amino acid sequence}', f'<FASTA>{smi[1]}</FASTA>'
                    )
                    # print(user_text)
                    # exit(1)
                elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                    user_text = tdc_prompts_json[TASK_NAME.replace('-', '_')].replace(
                        '{Peptide amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                    ).replace(
                        '{Possible MHC pseudosequences}', f'<FASTA>{smi[1]}</FASTA>'
                    )
                    # print(user_text)
                    # exit(1)
                else:
                    user_text = tdc_prompts_json[TASK_NAME.replace('-', '_')].replace(
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
            outs = llm.generate(base_prompts, sp, lora_request=lora_req)

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

            # ===================== 评估指标计算 =====================
            if len(probs) > 0:
                print("\n" + "="*80)
                print(f"{TASK_NAME} EVALUATION RESULTS (single-forward, next-token logprobs)")
                print("="*80)
                print('Score: p = P("(B)") / ( P("(A)") + P("(B)") ), using next-token logprobs (generation step)')
                print(f"Valid samples: {len(valid_idx)}/{len(base_prompts)}")

                # 计算AUROC
                if METRIC_TYPE in ["auroc", "both"]:
                    if len(set(valid_labels)) > 1:
                        auroc = roc_auc_score(valid_labels, probs)
                        print(f"AUROC: {auroc:.4f}")
                        if TASK_GROUP_NAME == 'Tox':
                            if TASK_NAME.startswith('herg_central'):
                                herg_central_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('ToxCast'):
                                ToxCast_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('Tox21'):
                                Tox21_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('Skin_Reaction'):
                                Skin_Reaction_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('hERG'):
                                hERG_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('AMES'):
                                AMES_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('DILI'):
                                DILI_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('ClinTox'):
                                ClinTox_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'ADME':
                            if TASK_NAME.startswith('PAMPA_NCATS'):
                                PAMPA_NCATS_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('HIA_Hou'):
                                HIA_Hou_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('Bioavailability_Ma'):
                                Bioavailability_Ma_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('BBB_Martins'):
                                BBB_Martins_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('Pgp_Broccatelli'):
                                Pgp_Broccatelli_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP1A2_Veith'):
                                CYP1A2_Veith_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP2C19_Veith'):
                                CYP2C19_Veith_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP2C9_Veith'):
                                CYP2C9_Veith_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP2D6_Veith'):
                                CYP2D6_Veith_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP3A4_Veith'):
                                CYP3A4_Veith_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP2C9_Substrate_CarbonMangels'):
                                CYP2C9_Substrate_CarbonMangels_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP2D6_Substrate_CarbonMangels'):
                                CYP2D6_Substrate_CarbonMangels_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('CYP3A4_Substrate_CarbonMangels'):
                                CYP3A4_Substrate_CarbonMangels_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'HTS':
                            if TASK_NAME.startswith('HIV'):
                                HIV_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('SARSCoV2_3CLPro_Diamond'):
                                SARSCoV2_3CLPro_Diamond_AUROCS.append(auroc)
                            elif TASK_NAME.startswith('SARSCoV2_Vitro_Touret'):
                                SARSCoV2_Vitro_Touret_AUROCS.append(auroc)
                            elif TASK_NAME.endswith('butkiewicz'):
                                butkiewicz_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'Develop':
                            # multi input sequences
                            if TASK_NAME.startswith('SAbDab_Chen'):
                                SAbDab_Chen_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'PPI':
                            # multi input sequences
                            if TASK_NAME.startswith('HuRI'):
                                HuRI_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'TCREpitopeBinding':
                            # multi input sequences
                            if TASK_NAME.startswith('Weber'):
                                Weber_AUROCS.append(auroc)
                        elif TASK_GROUP_NAME == 'PeptideMHC':
                            # multi input sequences
                            if TASK_NAME.startswith('MHC1_IEDB-IMGT_Nielsen'):
                                MHC1_IEDB_IMGT_Nielsen_AUROCS.append(auroc)
                            if TASK_NAME.startswith('MHC2_IEDB_Jensen'):
                                MHC2_IEDB_Jensen_AUROCS.append(auroc)
                    else:
                        print("Cannot compute AUROC: only one class in valid samples.")

                # 计算Accuracy
                if METRIC_TYPE in ["accuracy", "both"]:
                    # 根据阈值将概率转换为预测标签
                    predictions = [1 if p >= ACCURACY_THRESHOLD else 0 for p in probs]
                    acc = accuracy_score(valid_labels, predictions)
                    print(f"Accuracy (threshold={ACCURACY_THRESHOLD}): {acc:.4f}")

                    # 显示混淆矩阵信息
                    tp = sum(1 for pred, label in zip(predictions, valid_labels) if pred == 1 and label == 1)
                    tn = sum(1 for pred, label in zip(predictions, valid_labels) if pred == 0 and label == 0)
                    fp = sum(1 for pred, label in zip(predictions, valid_labels) if pred == 1 and label == 0)
                    fn = sum(1 for pred, label in zip(predictions, valid_labels) if pred == 0 and label == 1)
                    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

                print("="*80 + "\n")
            else:
                print("No valid samples for evaluation.")

            # （可选）调试前 3 个样本
            for j in range(min(3, len(valid_idx))):
                i = valid_idx[j]
                print(f"[Sample {i}] p={probs[j]:.4f}, y={valid_labels[j]}")

    print("="*80 + "\n")
    print(MODEL_NAME)
    for TASK_GROUP_NAME in TASK_GROUP_NAMEs:
        if TASK_GROUP_NAME == 'Tox':
            print('herg_central', np.mean(np.array(herg_central_AUROCS)))
            print('ToxCast', np.mean(np.array(ToxCast_AUROCS)))
            print('Tox21', np.mean(np.array(Tox21_AUROCS)))
            print('Skin_Reaction', np.mean(np.array(Skin_Reaction_AUROCS)))
            print('hERG', np.mean(np.array(hERG_AUROCS)))
            print('AMES', np.mean(np.array(AMES_AUROCS)))
            print('DILI', np.mean(np.array(DILI_AUROCS)))
            print('ClinTox', np.mean(np.array(ClinTox_AUROCS)))
        elif TASK_GROUP_NAME == 'ADME':
            print('PAMPA_NCATS', np.mean(np.array(PAMPA_NCATS_AUROCS)))
            print('HIA_Hou', np.mean(np.array(HIA_Hou_AUROCS)))
            print('Bioavailability_Ma', np.mean(np.array(Bioavailability_Ma_AUROCS)))
            print('BBB_Martins', np.mean(np.array(BBB_Martins_AUROCS)))
            print('Pgp_Broccatelli', np.mean(np.array(Pgp_Broccatelli_AUROCS)))
            print('CYP1A2_Veith', np.mean(np.array(CYP1A2_Veith_AUROCS)))
            print('CYP2C19_Veith', np.mean(np.array(CYP2C19_Veith_AUROCS)))
            print('CYP2C9_Veith', np.mean(np.array(CYP2C9_Veith_AUROCS)))
            print('CYP2D6_Veith', np.mean(np.array(CYP2D6_Veith_AUROCS)))
            print('CYP3A4_Veith', np.mean(np.array(CYP3A4_Veith_AUROCS)))
            print('CYP2C9_Substrate_CarbonMangels', np.mean(np.array(CYP2C9_Substrate_CarbonMangels_AUROCS)))
            print('CYP2D6_Substrate_CarbonMangels', np.mean(np.array(CYP2D6_Substrate_CarbonMangels_AUROCS)))
            print('CYP3A4_Substrate_CarbonMangels', np.mean(np.array(CYP3A4_Substrate_CarbonMangels_AUROCS)))
        elif TASK_GROUP_NAME == 'HTS':
            print('HIV', np.mean(np.array(HIV_AUROCS)))
            print('SARSCoV2_3CLPro_Diamond', np.mean(np.array(SARSCoV2_3CLPro_Diamond_AUROCS)))
            print('SARSCoV2_Vitro_Touret', np.mean(np.array(SARSCoV2_Vitro_Touret_AUROCS)))
            print('butkiewicz', np.mean(np.array(butkiewicz_AUROCS)))
        elif TASK_GROUP_NAME == 'Develop':
            print('SAbDab_Chen', np.mean(np.array(SAbDab_Chen_AUROCS)))
        elif TASK_GROUP_NAME == 'PPI':
            print('HuRI', np.mean(np.array(HuRI_AUROCS)))
        elif TASK_GROUP_NAME == 'TCREpitopeBinding':
            print('Weber', np.mean(np.array(Weber_AUROCS)))
        elif TASK_GROUP_NAME == 'PeptideMHC':
            print('MHC1_IEDB-IMGT_Nielsen', np.mean(np.array(MHC1_IEDB_IMGT_Nielsen_AUROCS)))
            print('MHC2_IEDB_Jensen', np.mean(np.array(MHC2_IEDB_Jensen_AUROCS)))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()