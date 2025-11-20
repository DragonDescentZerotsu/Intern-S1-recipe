'''
可以选择把数据保存成LlamaFactory需要的Alpaca格式，或者是Megatron需要的GPT格式
'''

import json
from huggingface_hub import hf_hub_download
# from tdc.single_pred import ADME, Tox, HTS, Develop, CRISPROutcome, Yields
# from tdc.multi_pred.ppi import PPI
# from tdc.multi_pred.tcr_epi import TCREpitopeBinding
# from tdc.multi_pred.trialoutcome import TrialOutcome
# from tdc.multi_pred.peptidemhc import PeptideMHC
# from tdc.multi_pred.dti import DTI
# from tdc.multi_pred.drugsyn import DrugSyn
# from tdc.multi_pred.drugres import DrugRes
# from tdc.multi_pred.antibodyaff import AntibodyAff
# from tdc.utils import retrieve_label_name_list
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import ast
from pathlib import Path
import numpy as np

current_dir = Path(__file__).parent.resolve()

# ===================== 配置 =====================
DATA_STYLE = 'Alpaca'  # 'GPT' or 'Alpaca'
SPLIT = 'test'  # 'train' or 'valid' or 'test'

RAW_DATA_DIR = current_dir / 'ChemCoTDataset-raw-data'
PROCESSED_DATA_DIR = current_dir / 'SFT_data' / (DATA_STYLE +'_ChemCoT')
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
if DATA_STYLE == 'Alpaca':
    OUTPUT_FILE = 'ChemCoT_SFT_data_all_tasks.json'   # 输出文件名
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
OUTPUT_FILE = PROCESSED_DATA_DIR / OUTPUT_FILE
INPUT_TYPE = "{Drug SMILES}"

model_name = "internlm/Intern-S1-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
ENABLE_THINKING = True

all_sft_data = []

fail_count = 0

# 定义所有任务组和任务
for file in RAW_DATA_DIR.glob('*.json'):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in tqdm(data, desc=f"Processing {file.name}"):
            user_text = sample['query']
            try:
                struct_cot = json.loads(sample['struct_cot'])
                cot_str = '\n'.join([key + ':\n' + str(value) for key, value in struct_cot.items()])
            except:
                fail_count += 1
                continue

            if file.name=='add.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: {json.loads(sample['meta'])['reference']}"

            elif file.name == 'delete.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: {json.loads(sample['meta'])['reference']}"

            elif file.name == 'drd.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'fg_count.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘count’: {json.loads(sample['meta'])['gt']},\n}}"

            elif file.name == 'fs_by_product.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘By Product’: {json.loads(sample['meta'])['gt']},\n}}"

            elif file.name == 'fs_major_product.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Major Product’: {json.loads(sample['meta'])['gt']},\n}}"

            elif file.name == 'gsk.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'jnk.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'logp.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'Murcko_scaffold.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Output Scaffold’: {json.loads(sample['meta'])['gt']},\n}}"

            elif file.name == 'qed.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'ring_count.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘count’: {json.loads(sample['meta'])['gt']},\n}}"

            elif file.name == 'ring_system_scaffold.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘output’: {struct_cot['output']},\n}}"

            elif file.name == 'solubility.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: \n{{\n    ‘Final Target Molecule’: {json.loads(sample['meta'])['reference']},\n}}"

            elif file.name == 'sub.json':
                output = f"<think>\n{cot_str}\n</think>\nAnswer: {json.loads(sample['meta'])['reference']}"

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

# ===================== 保存数据 =====================
print(f"\n{'='*80}")
print(f"Total SFT samples: {len(all_sft_data)}")
print(f"Fail count: {fail_count}")
print(f"Saving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    if DATA_STYLE == 'Alpaca':
        json.dump(all_sft_data, f, ensure_ascii=False, indent=2)
    elif DATA_STYLE == 'GPT':
        for item in tqdm(all_sft_data, desc=f'Saving {DATA_STYLE} SFT data'):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
print(f"Done! Data saved to {OUTPUT_FILE}")
print(f"{'='*80}\n")