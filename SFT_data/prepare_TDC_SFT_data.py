'''
可以选择把数据保存成LlamaFactory需要的Alpaca格式，或者是Megatron需要的GPT格式
'''

import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox, HTS, Develop, CRISPROutcome, Yields
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.multi_pred.dti import DTI
from tdc.multi_pred.drugsyn import DrugSyn
from tdc.multi_pred.drugres import DrugRes
from tdc.multi_pred.antibodyaff import AntibodyAff
from tdc.utils import retrieve_label_name_list
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import ast
from pathlib import Path
import numpy as np

current_dir = Path(__file__).parent.resolve()

# ===================== 配置 =====================
DATA_STYLE = 'GPT'  # 'GPT' or 'Alpaca'
SPLIT = 'test'  # 'train' or 'valid' or 'test'

DATA_DIR = current_dir / 'SFT_data' / (DATA_STYLE +'_TDC_CLS')
DATA_DIR.mkdir(exist_ok=True)
if DATA_STYLE == 'Alpaca':
    OUTPUT_FILE = 'TDC_SFT_data_all_tasks.json'   # 输出文件名
elif DATA_STYLE == 'GPT':
    if SPLIT == 'train':
        OUTPUT_FILE = 'training.jsonl'
    elif SPLIT == 'valid':
        OUTPUT_FILE = 'validation.jsonl'
    elif SPLIT == 'test':
        OUTPUT_FILE = 'test.jsonl'
    else:
        print("Wrong SPLIT: Not 'train', 'valid' or 'test'")
        exit(1)
OUTPUT_FILE = DATA_DIR / OUTPUT_FILE
INPUT_TYPE = "{Drug SMILES}"

model_name = "internlm/Intern-S1-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
ENABLE_THINKING = False

# 定义所有任务组和任务
ALL_TASK_CONFIGS = {
    'Tox': (
        ["hERG_Karim", "Carcinogens_Lagunin"] +  # Accuracy
        ['Skin_Reaction', 'hERG', 'AMES', 'DILI', 'ClinTox'] +
        ['Tox21'+'_'+label.replace('-', '_') for label in retrieve_label_name_list('Tox21')] +
        ['herg_central'+'_'+retrieve_label_name_list('herg_central')[-1]] +
        ['ToxCast'+'_'+label for label in retrieve_label_name_list('Toxcast')]
    ),
    'ADME': [
        'PAMPA_NCATS', 'HIA_Hou', 'Bioavailability_Ma', 'BBB_Martins', 'Pgp_Broccatelli',
        'CYP1A2_Veith', 'CYP2C19_Veith', 'CYP2C9_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith',
        'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels',
    ],
    'HTS': [
        'cav3_t-type_calcium_channels_butkiewicz', 'tyrosyl-dna_phosphodiesterase_butkiewicz',
        'HIV', 'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret',
        'orexin1_receptor_butkiewicz', 'm1_muscarinic_receptor_agonists_butkiewicz',
        'm1_muscarinic_receptor_antagonists_butkiewicz', 'potassium_ion_channel_kir2.1_butkiewicz',
        'kcnq2_potassium_channel_butkiewicz', 'choline_transporter_butkiewicz',
        'serine_threonine_kinase_33_butkiewicz',
    ],
    'Develop': ['SAbDab_Chen'],
    'PPI': ['HuRI'],
    'TCREpitopeBinding': ['Weber'],
    # 'TrialOutcome': ['phase1', 'phase2', 'phase3'],
    # 'PeptideMHC': ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']  # TODO: 这个有点问题，好像不是二分类？/some problems, maybe not binary cls?
}

# ===================== 加载TDC模板 =====================
tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-predict",
    filename="tdc_prompts.json",
)
tdc_prompts_json = json.load(open(tdc_prompts_filepath, "r"))
# fix a key bug
tdc_prompts_json['SARSCoV2_3CLPro_Diamond'] = tdc_prompts_json['SARSCOV2_3CLPro_Diamond']

# ===================== 生成SFT数据 =====================
all_sft_data = []

# 遍历所有 classification 任务组
for TASK_GROUP_NAME, TASK_NAMEs in tqdm(ALL_TASK_CONFIGS.items(), desc="Processing task groups"):
    print(f"\n{'='*80}")
    print(f"Processing {TASK_GROUP_NAME} tasks...")
    print(f"{'='*80}")

    # 遍历该任务组下的所有任务
    for TASK_NAME in tqdm(TASK_NAMEs, desc=f"Processing {TASK_GROUP_NAME} tasks", leave=False):
        print(f"\nProcessing {TASK_NAME}...")

        # 加载数据
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

        # 获取训练集数据
        split = data.get_split()

        # 提取输入序列
        if TASK_NAME == 'SAbDab_Chen':
            train_drugs = split[SPLIT]['Antibody'].values
        elif TASK_NAME == 'HuRI':
            train_drugs = split[SPLIT][['Protein1', 'Protein2']].values
        elif TASK_NAME == 'Weber':
            # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
            train_drugs = split[SPLIT][['epitope_aa', 'tcr']].values
        elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
            # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
            train_drugs = split[SPLIT][['Peptide', 'MHC']].values
        else:
            train_drugs = split[SPLIT]['Drug'].values

        # 提取标签
        if TASK_NAME in ['Weber']:
            train_labels = split[SPLIT]['label'].values
        else:
            train_labels = split[SPLIT]['Y'].values  # 0/1

        print(f"Total {SPLIT} samples: {len(train_drugs)}")

        # 生成Alpaca格式的数据
        for smi, label in zip(train_drugs, train_labels):
            # 处理特殊任务名称
            task_name_for_prompt = TASK_NAME
            if TASK_NAME.startswith('herg_central'):
                task_name_for_prompt = 'herg_central'

            # 根据不同任务类型生成instruction
            if TASK_NAME == 'SAbDab_Chen':
                smi = ast.literal_eval(smi)
                user_text = tdc_prompts_json[task_name_for_prompt.replace('-', '_')].replace(
                    '{Antibody heavy chain sequence}', f'<FASTA>{smi[0]}</FASTA>'
                ).replace(
                    '{Antibody light chain sequence}', f'<FASTA>{smi[1]}</FASTA>'
                )
            elif TASK_NAME == 'HuRI':
                user_text = tdc_prompts_json[task_name_for_prompt.replace('-', '_')].replace(
                    '{Protein1 amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                ).replace(
                    '{Protein2 amino acid sequence}', f'<FASTA>{smi[1]}</FASTA>'
                )
            elif TASK_NAME == 'Weber':
                user_text = tdc_prompts_json['weber'].replace(
                    '{Epitope amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                ).replace(
                    '{TCR amino acid sequence}', f'<FASTA>{smi[1]}</FASTA>'
                )
            elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
                user_text = tdc_prompts_json[task_name_for_prompt.replace('-', '_')].replace(
                    '{Peptide amino acid sequence}', f'<FASTA>{smi[0]}</FASTA>'
                ).replace(
                    '{Possible MHC pseudosequences}', f'<FASTA>{smi[1]}</FASTA>'
                )
            else:
                user_text = tdc_prompts_json[task_name_for_prompt.replace('-', '_')].replace(
                    INPUT_TYPE, f'<SMILES>{smi}</SMILES>'
                )

            # 根据标签生成output：0->(A), 1->(B)
            if label == 1:
                output = "(B)"
            elif label == 0:
                output = "(A)"
            else:
                print("WRONG LABEL: Not 1 or 0")
                print(f'Wrong Label: {label}')
                exit(1)

            # 创建Alpaca格式的数据
            if DATA_STYLE == 'Alpaca':
                alpaca_data = {
                    "instruction": user_text,
                    "input": "",
                    "output": output
                }

                all_sft_data.append(alpaca_data)
            elif DATA_STYLE == 'GPT':
                message = [
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": output}
                    ]
                rendered = tokenizer.apply_chat_template(
                            message,
                            tokenize=False,
                            add_generation_prompt=False,  # 训练样本末尾不加起始标记
                            enable_thinking=ENABLE_THINKING  # 想开“思考模式”就改成 True
                        )
                input = rendered.split("assistant")[0] + "assistant"
                answer_text = rendered.split("assistant")[1]
                gpt_data = {"input": input, "output": answer_text}

                all_sft_data.append(gpt_data)


# ------------ Regression -----------------
# def _scale_to_0_1000(arr, y_min=None, y_max=None):
#     """按训练集 min-max 线性缩放到整数 [0, 1000]。"""
#     arr = np.asarray(arr, dtype=float)
#     y_min = float(np.min(arr) if y_min is None else y_min)
#     y_max = float(np.max(arr) if y_max is None else y_max)
#     if y_max == y_min:
#         scaled = np.zeros_like(arr, dtype=int)
#     else:
#         scaled = np.round((arr - y_min) / (y_max - y_min) * 1000).astype(int)
#     scaled = np.clip(scaled, 0, 1000)
#     return scaled, y_min, y_max
#
# def _fmt_000(n: int) -> str:
#     """把整数格式化为 '000'..'999' 或 '1000'。"""
#     return "1000" if int(n) >= 1000 else f"{int(n):03d}"
#
# def _safe_get(df, candidates, default=""):
#     for c in candidates:
#         if c in df.columns:
#             return df[c].values
#     return np.array([default] * len(df))
#
# print(f"\n{'='*80}")
# print("Appending regression tasks...")
# print(f"{'='*80}")
#
# # ---------- 1) ADME / 物化性质（单输入 SMILES） ----------
# ADME_REG_TASKS = [
#     'Caco2_Wang',
#     'VDss_Lombardo', 'Half_Life_Obach', 'PPBR_AZ',
#     'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ',
#     # 'LD50_Zhu',
#     'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB',
# ]
# for TASK_NAME in ADME_REG_TASKS:
#     print(f"Regression-ADME: {TASK_NAME}")
#     data = ADME(name=TASK_NAME)  # 见 TDC ADME 文档
#     split = data.get_split()
#     df = split['train']
#     xs = df['Drug'].values
#     ys = df['Y'].values.astype(float)
#     ys_scaled, y_min, y_max = _scale_to_0_1000(ys)
#
#     prompt_key = TASK_NAME.replace('-', '_')
#     for smi, y_s in zip(xs, ys_scaled):
#         user_text = tdc_prompts_json[prompt_key].replace(
#             '{Drug SMILES}', f'<SMILES>{smi}</SMILES>'
#         )
#         all_sft_data.append({
#             "instruction": user_text,
#             "input": "",
#             "output": _fmt_000(int(y_s))
#         })
#
# # ---------- 2) DTI（SMILES + 蛋白 FASTA） ----------
# DTI_TASKS = ['BindingDB_Kd', 'BindingDB_IC50', 'BindingDB_Ki', 'DAVIS', 'KIBA']
# DTI_PROMPT_KEY = {
#     'BindingDB_Kd': 'BindingDB_kd',
#     'BindingDB_IC50': 'BindingDB_ic50',
#     'BindingDB_Ki': 'BindingDB_ki',
#     'DAVIS': 'DAVIS',
#     'KIBA': 'KIBA',
# }
# for TASK_NAME in DTI_TASKS:
#     print(f"Regression-DTI: {TASK_NAME}")
#     data = DTI(name=TASK_NAME)  # 见 TDC DTI 文档
#     split = data.get_split()
#     df = split['train']
#     drug = df['Drug'].values
#     target = df['Target'].values
#     ys_scaled, _, _ = _scale_to_0_1000(df['Y'].values.astype(float))
#
#     pkey = DTI_PROMPT_KEY[TASK_NAME]
#     for smi, seq, y_s in zip(drug, target, ys_scaled):
#         user_text = tdc_prompts_json[pkey] \
#             .replace('{Drug SMILES}', f'<SMILES>{smi}</SMILES>') \
#             .replace('{Target amino acid sequence}', f'<FASTA>{seq}</FASTA>')
#         all_sft_data.append({
#             "instruction": user_text,
#             "input": "",
#             "output": _fmt_000(int(y_s))
#         })
#
# # ---------- 3) Drug Synergy（两条 SMILES + 细胞系） ----------
# # 3.1 OncoPolyPharmacology（单一 Y）
# print("Regression-DrugSyn: OncoPolyPharmacology")
# syn_data = DrugSyn(name='OncoPolyPharmacology')  # 见 DrugSyn 文档
# syn_split = syn_data.get_split()
# syn_train = syn_split['train']
# ys_scaled, _, _ = _scale_to_0_1000(syn_train['Y'].values.astype(float))
# cell = syn_train['Cell_Line'].values
# for d1, d2, ce, y_s in zip(syn_train['Drug1'].values, syn_train['Drug2'].values, cell, ys_scaled):
#     user_text = tdc_prompts_json['OncoPolyPharmacology'] \
#         .replace('{Drug1 SMILES}', f'<SMILES>{d1}</SMILES>') \
#         .replace('{Drug2 SMILES}', f'<SMILES>{d2}</SMILES>') \
#         .replace('{Cell line description}', str(ce))
#     all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # 3.2 DrugComb（五个指标：CSS/HSA/Loewe/Bliss/ZIP）
# print("Regression-DrugSyn: DrugComb (CSS/HSA/Loewe/Bliss/ZIP)")
# comb_data = DrugSyn(name='DrugComb')  # 见 DrugSyn 文档
# comb_split = comb_data.get_split()
# comb_train = comb_split['train']
# # 部分版本直接给出各列，也可能只有 'Y'；尽量兼容：
# comb_metrics = ['CSS', 'Synergy_HSA', 'Synergy_Loewe', 'Synergy_Bliss', 'Synergy_ZIP']
# cell = _safe_get(comb_train, ['Cell line', 'Cell Line', 'CellLine', 'Cell', 'cell_line'], "")
# for metric in comb_metrics:
#     if metric in comb_train.columns:
#         y_col = comb_train[metric].values.astype(float)
#     elif 'Y' in comb_train.columns:
#         # 如果只有单列 Y，就复用这列（仍然会按该列 min-max 到 0-1000）
#         y_col = comb_train['Y'].values.astype(float)
#     else:
#         continue
#     ys_scaled, _, _ = _scale_to_0_1000(y_col)
#     prompt_key = f'DrugComb_{metric}'.replace('_Synergy_', '_')
#     for d1, d2, ce, y_s in tqdm(zip(comb_train['Drug1'].values, comb_train['Drug2'].values, cell, ys_scaled), total=len(ys_scaled), desc='Processing prompts'):
#         user_text = tdc_prompts_json[prompt_key] \
#             .replace('{Drug1 SMILES}', f'<SMILES>{d1}</SMILES>') \
#             .replace('{Drug2 SMILES}', f'<SMILES>{d2}</SMILES>') \
#             .replace('{Cell line description}', str(ce))
#         all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # ---------- 4) Drug Response（GDSC1 / GDSC2：SMILES + 细胞系） ----------
# for TASK_NAME in ['GDSC1', 'GDSC2']:
#     print(f"Regression-DrugRes: {TASK_NAME}")
#     data = DrugRes(name=TASK_NAME)  # 见 DrugRes 文档
#     split = data.get_split()
#     df = split['train']
#     ys_scaled, _, _ = _scale_to_0_1000(df['Y'].values.astype(float))
#     cell = _safe_get(df, ['Cell line', 'Cell Line', 'CellLine', 'Cell', 'cell_line', 'cell_name'], "")
#     for smi, ce, y_s in tqdm(zip(df['Drug'].values, cell, ys_scaled), total=len(ys_scaled), desc='Processing prompts'):
#         user_text = tdc_prompts_json[TASK_NAME] \
#             .replace('{Drug SMILES}', f'<SMILES>{smi}</SMILES>') \
#             .replace('{Cell line description}', str(ce))
#         all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # ---------- 5) 蛋白-抗体亲和力（Protein_SAbDab：抗原 + 重链 + 轻链） ----------
# print("Regression-AntibodyAff: Protein_SAbDab")
# aa_data = AntibodyAff(name='Protein_SAbDab')  # 见 AntibodyAff 文档
# aa_split = aa_data.get_split()
# aa_train = aa_split['train']
# ys_scaled, _, _ = _scale_to_0_1000(aa_train['Y'].values.astype(float))
# for ab, ag, y_s in zip(aa_train['Antibody'].values, aa_train['Antigen'].values, ys_scaled):
#     # Antibody 字段通常是 "('heavy','light')" 的字符串
#     try:
#         heavy, light = ast.literal_eval(ab)
#     except Exception:
#         # 如果已经是 tuple/list 或其它格式，尽力兼容
#         if isinstance(ab, (list, tuple)) and len(ab) >= 2:
#             heavy, light = ab[0], ab[1]
#         else:
#             heavy, light = ab, ''
#     user_text = tdc_prompts_json['Protein_SAbDab'] \
#         .replace('{Antigen sequence}', f'<FASTA>{ag}</FASTA>') \
#         .replace('{Antibody heavy chain sequence}', f'<FASTA>{heavy}</FASTA>') \
#         .replace('{Antibody light chain sequence}', f'<FASTA>{light}</FASTA>')
#     all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # ---------- 6) 抗体开发性（TAP：五个 label_name） ----------
# print("Regression-Develop: TAP (CDR_Length / PSH / PPC / PNC / SFvCSP)")
# tap_labels = retrieve_label_name_list('TAP')  # 见 TAP 文档
# # 期望得到类似：['CDR_Length','PSH','PPC','PNC','SFvCSP']
# TAP_NAME_MAP = {
#     'TAP_CDR_Length': 'CDR_Length',
#     'TAP_PSH': 'PSH',
#     'TAP_PPC': 'PPC',
#     'TAP_PNC': 'PNC',
#     'TAP_SFvCSP': 'SFvCSP',
# }
# for task_name, label_name in TAP_NAME_MAP.items():
#     # 容错：如果官方返回大小写/命名有差异，尽量匹配
#     match = [l for l in tap_labels if l.lower() == label_name.lower()]
#     label = match[0] if match else tap_labels[0]
#     tap = Develop(name='TAP', label_name=label)
#     split = tap.get_split()
#     df = split['train']
#     ys_scaled, _, _ = _scale_to_0_1000(df['Y'].values.astype(float))
#     for ab, y_s in zip(df['Antibody'].values, ys_scaled):
#         try:
#             heavy, light = ast.literal_eval(ab)
#         except Exception:
#             if isinstance(ab, (list, tuple)) and len(ab) >= 2:
#                 heavy, light = ab[0], ab[1]
#             else:
#                 heavy, light = ab, ''
#         user_text = tdc_prompts_json[task_name] \
#             .replace('{Antibody heavy chain sequence}', f'<FASTA>{heavy}</FASTA>') \
#             .replace('{Antibody light chain sequence}', f'<FASTA>{light}</FASTA>')
#         all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # ---------- 7) CRISPR Repair Outcome（Leenay：五个 label_name） ----------
# print("Regression-CRISPROutcome: Leenay (5 labels)")
# leenay_labels = retrieve_label_name_list('Leenay')  # 见 Leenay 文档
# # 关键词到真实 label 的模糊匹配
# LEENAY_MAP = {
#     'Leenay_Fraction_Insertions': ['fraction', 'insertion'],
#     'Leenay_Avg_Insertion_Length': ['avg', 'insertion', 'length'],
#     'Leenay_Avg_Deletion_Length': ['avg', 'deletion', 'length'],
#     'Leenay_Indel_Diversity': ['diversity'],
#     'Leenay_Fraction_Frameshifts': ['fraction', 'frameshift'],
# }
# def _pick_label(name_keywords):
#     for cand in leenay_labels:
#         if all(k.lower() in cand.lower() for k in name_keywords):
#             return cand
#     return leenay_labels[0]
# for task_name, keys in LEENAY_MAP.items():
#     label = _pick_label(keys)
#     data = CRISPROutcome(name='Leenay', label_name=label)
#     split = data.get_split()
#     df = split['train']
#     ys_scaled, _, _ = _scale_to_0_1000(df['Y'].values.astype(float))
#     for gseq, y_s in zip(df['Drug'].values, ys_scaled):   # 该任务把序列放在 'Drug' 列
#         user_text = tdc_prompts_json[task_name] \
#             .replace('{GuideSeq}', f'<FASTA>{gseq}</FASTA>')
#         all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})
#
# # ---------- 8) 反应收率（Buchwald-Hartwig / USPTO_Yields） ----------
# for TASK_NAME in ['Buchwald-Hartwig', 'USPTO_Yields']:
#     print(f"Regression-Yields: {TASK_NAME}")
#     ydata = Yields(name=TASK_NAME)  # 见 Yields 文档
#     split = ydata.get_split()
#     df = split['train']
#     ys_scaled, _, _ = _scale_to_0_1000(df['Y'].values.astype(float))
#     # 常见列名：Catalyst / Reactant(s) / Product
#     catalyst = _safe_get(df, ['Catalyst', 'Catalyst_SMILES', 'catalyst'], "")
#     product = _safe_get(df, ['Product', 'Products', 'product'], "")
#     # 反应物可能有多个，尽量拼成一个用 '.' 连接
#     reactant_cols = [c for c in df.columns if c.lower().startswith('reactant')]
#     if not reactant_cols and 'Reactant' in df.columns:
#         reactant_cols = ['Reactant']
#     if reactant_cols:
#         reactant_joined = df[reactant_cols].astype(str).agg('.'.join, axis=1).values
#     else:
#         reactant_joined = _safe_get(df, ['Reactants', 'Reactant', 'reactant'], "")
#     pkey = 'Buchwald_Hartwig' if TASK_NAME == 'Buchwald-Hartwig' else 'USPTO_Yields'
#     for cat, rea, prod, y_s in zip(catalyst, reactant_joined, product, ys_scaled):
#         user_text = tdc_prompts_json[pkey] \
#             .replace('{Catalyst SMILES}', f'<SMILES>{cat}</SMILES>') \
#             .replace('{Reactant SMILES}', f'<SMILES>{rea}</SMILES>') \
#             .replace('{Product SMILES}', f'<SMILES>{prod}</SMILES>')
#         all_sft_data.append({"instruction": user_text, "input": "", "output": _fmt_000(int(y_s))})

# ===================== 保存数据 =====================
print(f"\n{'='*80}")
print(f"Total SFT samples: {len(all_sft_data)}")
print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    if DATA_STYLE == 'Alpaca':
        json.dump(all_sft_data, f, ensure_ascii=False, indent=2)
    elif DATA_STYLE == 'GPT':
        for item in tqdm(all_sft_data, desc=f'Saving {DATA_STYLE} SFT data'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Done! Data saved to {OUTPUT_FILE}")
print(f"{'='*80}\n")