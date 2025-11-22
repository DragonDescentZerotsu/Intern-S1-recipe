import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME, Tox, HTS, Develop
from tdc.multi_pred.ppi import PPI
from tdc.multi_pred.tcr_epi import TCREpitopeBinding
from tdc.multi_pred.trialoutcome import TrialOutcome
from tdc.multi_pred.peptidemhc import PeptideMHC
from tdc.utils import retrieve_label_name_list
from tqdm import tqdm
import ast
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

# ===================== 配置 =====================
DATA_DIR = current_dir / 'SFT_data'
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = 'TDC_SFT_data_all_tasks.json'  # 输出文件名
OUTPUT_FILE = DATA_DIR / OUTPUT_FILE
INPUT_TYPE = "{Drug SMILES}"

# 定义所有任务组和任务
ALL_TASK_CONFIGS = {
    'Tox': (
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
    'PeptideMHC': ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen'],
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

# 遍历所有任务组
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
            train_drugs = split['train']['Antibody'].values
        elif TASK_NAME == 'HuRI':
            train_drugs = split['train'][['Protein1', 'Protein2']].values
        elif TASK_NAME == 'Weber':
            # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
            train_drugs = split['train'][['epitope_aa', 'tcr']].values
        elif TASK_NAME in ['MHC1_IEDB-IMGT_Nielsen', 'MHC2_IEDB_Jensen']:
            # take amino acid sequence as input, although they offer SMILES. In the TxGemma template, they use aa seqs.
            train_drugs = split['train'][['Peptide', 'MHC']].values
        else:
            train_drugs = split['train']['Drug'].values

        # 提取标签
        if TASK_NAME in ['Weber']:
            train_labels = split['train']['label'].values
        else:
            train_labels = split['train']['Y'].values  # 0/1

        print(f"Total train samples: {len(train_drugs)}")

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
            alpaca_data = {
                "instruction": user_text,
                "input": "",
                "output": output
            }

            all_sft_data.append(alpaca_data)

# ===================== 保存数据 =====================
print(f"\n{'='*80}")
print(f"Total SFT samples: {len(all_sft_data)}")
print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(all_sft_data, f, ensure_ascii=False, indent=2)
print(f"Done! Data saved to {OUTPUT_FILE}")
print(f"{'='*80}\n")