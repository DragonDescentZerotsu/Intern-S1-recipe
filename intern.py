from transformers import AutoProcessor, AutoModelForCausalLM
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME
from sklearn.metrics import f1_score
import re
from tqdm import tqdm

def extract_answer(response):
    """从模型回答中提取最后一个Answer:之后的内容"""
    # 找到所有"Answer:"的位置
    answer_matches = list(re.finditer(r'Answer:', response, re.IGNORECASE))
    if not answer_matches:
        return None

    # 获取最后一个"Answer:"之后的内容
    last_answer_pos = answer_matches[-1].end()
    answer_text = response[last_answer_pos:].strip()

    return answer_text

def parse_answer_v1(answer_text):
    """版本1: (A) -> 1, (B) -> 0"""
    if answer_text is None:
        return None
    if '(A)' in answer_text:
        return 1
    elif '(B)' in answer_text:
        return 0
    else:
        return None

def parse_answer_v2(answer_text):
    """版本2: (A) -> 0, (B) -> 1"""
    if answer_text is None:
        return None
    if '(A)' in answer_text:
        return 0
    elif '(B)' in answer_text:
        return 1
    else:
        return None

model_name = "internlm/Intern-S1-mini-FP8"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "text", "text": "tell me about an interesting physical phenomenon."},
#         ],
#     }
# ]

tdc_prompts_filepath = hf_hub_download(
    repo_id="google/txgemma-9b-predict",
    filename="tdc_prompts.json",
)
tdc_prompts_json = json.load(open(tdc_prompts_filepath))

# 选择一个任务和输入
task_name = "BBB_Martins"
input_type = "{Drug SMILES}"

data = ADME(name=task_name)
split = data.get_split()
test_drugs = split['test']['Drug']
test_labels = split['test']['Y']

print(f'Total test samples: {len(test_drugs)}')
print(f'Test labels: {test_labels[:5]}...')  # 显示前5个标签

# 存储两个版本的预测结果
predictions_v1 = []
predictions_v2 = []

# 循环处理所有测试样本
for idx, drug_smiles in tqdm(enumerate(test_drugs)):
    print(f'\n[{idx+1}/{len(test_drugs)}] Processing drug: {drug_smiles[:50]}...')

    # 生成prompt
    prompt = tdc_prompts_json[task_name].replace(input_type, drug_smiles)

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 处理输入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,  # TODO: True
        return_dict=True,
        return_tensors="pt"
    )#.to(model.device, dtype=torch.bfloat16)

    print(f'Model input: {inputs}')
    exit(0)
    # 生成回答
    generate_ids = model.generate(**inputs, max_new_tokens=32768)
    decoded_output = processor.decode(
        generate_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print(f'Model output: {decoded_output}')

    # 提取并解析答案
    answer_text = extract_answer(decoded_output)
    pred_v1 = parse_answer_v1(answer_text)
    pred_v2 = parse_answer_v2(answer_text)

    predictions_v1.append(pred_v1)
    predictions_v2.append(pred_v2)

    print(f'Extracted answer: {answer_text}')
    print(f'Prediction V1 (A->1, B->0): {pred_v1}')
    print(f'Prediction V2 (A->0, B->1): {pred_v2}')
    print(f'True label: {test_labels[idx]}')

# 处理None值：过滤掉无法解析的样本
valid_indices_v1 = [i for i, pred in enumerate(predictions_v1) if pred is not None]
valid_indices_v2 = [i for i, pred in enumerate(predictions_v2) if pred is not None]

filtered_predictions_v1 = [predictions_v1[i] for i in valid_indices_v1]
filtered_labels_v1 = [test_labels[i] for i in valid_indices_v1]

filtered_predictions_v2 = [predictions_v2[i] for i in valid_indices_v2]
filtered_labels_v2 = [test_labels[i] for i in valid_indices_v2]

print('\n' + '='*80)
print('EVALUATION RESULTS')
print('='*80)

# 计算F1 score - 版本1
if len(filtered_predictions_v1) > 0:
    f1_v1 = f1_score(filtered_labels_v1, filtered_predictions_v1, average='binary')
    print(f'\nVersion 1 (A->1, B->0):')
    print(f'  Valid samples: {len(filtered_predictions_v1)}/{len(predictions_v1)}')
    print(f'  F1 Score: {f1_v1:.4f}')
else:
    print(f'\nVersion 1 (A->1, B->0):')
    print(f'  No valid predictions!')

# 计算F1 score - 版本2
if len(filtered_predictions_v2) > 0:
    f1_v2 = f1_score(filtered_labels_v2, filtered_predictions_v2, average='binary')
    print(f'\nVersion 2 (A->0, B->1):')
    print(f'  Valid samples: {len(filtered_predictions_v2)}/{len(predictions_v2)}')
    print(f'  F1 Score: {f1_v2:.4f}')
else:
    print(f'\nVersion 2 (A->0, B->1):')
    print(f'  No valid predictions!')

print('\n' + '='*80)

# 确定哪个版本更好
if len(filtered_predictions_v1) > 0 and len(filtered_predictions_v2) > 0:
    if f1_v1 > f1_v2:
        print(f'\nBest mapping: Version 1 (A->1, B->0) with F1={f1_v1:.4f}')
    elif f1_v2 > f1_v1:
        print(f'\nBest mapping: Version 2 (A->0, B->1) with F1={f1_v2:.4f}')
    else:
        print(f'\nBoth versions have the same F1 score: {f1_v1:.4f}')
