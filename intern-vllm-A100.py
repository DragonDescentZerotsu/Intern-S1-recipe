import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import torch
import os
import json
from huggingface_hub import hf_hub_download
from tdc.single_pred import ADME
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

def extract_answer(response):
    """从模型回答中提取最后一个Answer:之后的内容"""
    # 找到所有"Answer:"的位置
    format_correct = False
    if 'Answer:' in response:
        format_correct = True
        answer_matches = list(re.finditer(r'Answer:', response, re.IGNORECASE))
        if not answer_matches:
            return None, format_correct
    elif 'answer is' in response:
        format_correct = True
        answer_matches = list(re.finditer(r'answer is', response, re.IGNORECASE))
        if not answer_matches:
            return None, format_correct

    # 获取最后一个"Answer:"之后的内容
    if 'Answer:' in response or 'answer is' in response:
        last_answer_pos = answer_matches[-1].end()
        answer_text = response[last_answer_pos:].strip()
    else:
        answer_text = response

    return answer_text, format_correct

def parse_answer(answer_text, format_correct):
    """解析答案: (A) -> 0 (负类), (B) -> 1 (正类)"""
    if answer_text is None:
        return None
    if format_correct:
        if '(A)' in answer_text:
            return 0
        elif 'A**' in answer_text:
            return 0
        elif 'A)' in answer_text:
            return 0
        elif '\\boxed{A}' in answer_text:
            return 0
        elif '\\text{A}' in answer_text:
            return 0
        elif 'A' in answer_text:
            return 0
        elif '(B)' in answer_text:
            return 1
        elif 'B**' in answer_text:
            return 1
        elif 'B)' in answer_text:
            return 1
        elif '\\boxed{B}' in answer_text:
            return 1
        elif '\\text{B}' in answer_text:
            return 1
        elif 'B' in answer_text:
            return 1
        else:
            return None
    else:
        if '\\boxed{A}' in answer_text:
            return 0
        elif '\\text{A}' in answer_text:
            return 0
        elif '\n(A)' in answer_text:
            return 0
        elif '\\boxed{B}' in answer_text:
            return 1
        elif '\\text{B}' in answer_text:
            return 1
        elif '\n(B)' in answer_text:
            return 1
        else:
            return None
# os.environ["MKL_THREADING_LAYER"] = "GNU"
# os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)
# # 可选
# os.environ.setdefault("OMP_NUM_THREADS", "1")

def main():

    fig_save_path = current_dir / 'figs'
    fig_save_path.mkdir(exist_ok=True)

    model_name = "internlm/Intern-S1-mini"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    llm = LLM(
        model=model_name,
        # tokenizer=model_name,
        enforce_eager=True,  # TODO: * this is very important
        # tokenizer_mode="slow",
        max_model_len=1024*8,
        max_num_batched_tokens=1024*8,  # 限制每轮处理 token 上限，减少瞬时显存
        # quantization="fp8", #"fp8",          # 触发 FP8 W8A8 路径（或自动识别 FP8 检查点）
        dtype="bfloat16",            # 非 FP8 运算与累加精度（与 FP8 内核配套）
        tensor_parallel_size=4 if model_name=="internlm/Intern-S1-FP8" else 1,      # Tensor Parallel（按你的卡数调整）
        # kv_cache_dtype="fp8_e5m2",   # H100 上常用的 KV Cache 精度
        trust_remote_code=True,
        gpu_memory_utilization=0.92,  # 0.85
        max_num_seqs=128,
        limit_mm_per_prompt={"video": 0, "image": 0}
        # 如果你的版本支持，可开启 KV 动态缩放（否则需从检查点提供缩放因子）
        # calculate_kv_scales=True,
    )

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
    test_drugs = split['test']['Drug']#[:5]
    test_labels = split['test']['Y']#[:5]

    print(f'Total test samples: {len(test_drugs)}')
    print(f'Test labels: {test_labels[:5]}...')  # 显示前5个标签

    def to_prompt(text: str) -> str:
        conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        return tokenizer.apply_chat_template(
            conv,
            tokenize=False,  # 让 vLLM 自己分词
            add_generation_prompt=True,  # 补上assistant起始标记
            enable_thinking=False
        )

    # 准备所有prompts
    all_prompts = []
    for drug_smiles in test_drugs:
        prompt_text = tdc_prompts_json[task_name].replace(input_type, f'<SMILES>{drug_smiles}</SMILES>')
        prompt_text = prompt_text.replace('Answer:', 'Please think step by step and then put your final choice after "Answer:"')
        all_prompts.append(to_prompt(prompt_text))

    print(f'\nGenerating predictions for {len(all_prompts)} samples...')

    # 批量生成
    sp = SamplingParams(max_tokens=1024*8,
                        temperature=0.7,
                        top_p=1.0,
                        n=16,
                        top_k=50)
    outputs = llm.generate(all_prompts, sp)

    # 存储预测结果
    predictions = []
    failed_parses = []  # 存储解析失败的样本

    # —— 聚合到每题的概率 \hat p ——（丢弃解析失败的样本）
    p_scores = []  # 每题的 \hat p
    per_item_stats = []  # 记录各题 A/B/None 计数

    print('\nProcessing sampled outputs...')
    for idx, out in enumerate(tqdm(outputs)):
        cnt_a, cnt_b, cnt_none = 0, 0, 0
        # vLLM: out.outputs 是长度 n 的候选列表
        for j, cand in enumerate(out.outputs):
            txt = cand.text.strip()
            ans_txt, fmt_ok = extract_answer(txt)
            pred = parse_answer(ans_txt, fmt_ok)
            if pred == 0:
                cnt_a += 1
            elif pred == 1:
                cnt_b += 1
            else:
                cnt_none += 1

            if pred is None:
                failed_parses.append({
                    'index': idx,
                    'drug_smiles': test_drugs[idx],
                    'full_output': txt,
                    'extracted_answer': ans_txt,
                    'true_label': test_labels[idx]
                })

            # 仅前2题展示前若干采样，便于 sanity check
            if idx < 2 and j < 3:
                print(f'\n[Item {idx} | Sample {j}]')
                print(f'Raw: {txt}')
                print(f'Parsed -> {pred}')


        denom = cnt_a + cnt_b
        if denom == 0:
            p_hat = None  # 全部解析失败，留空
        else:
            p_hat = cnt_b / denom

        p_scores.append(p_hat)
        per_item_stats.append((cnt_a, cnt_b, cnt_none))
        # 提取并解析答案
        # answer_text, format_correct = extract_answer(decoded_output)
        # pred = parse_answer(answer_text, format_correct)

        # predictions.append(pred)

        # 记录解析失败的样本
        # if pred is None:
        #     failed_parses.append({
        #         'index': idx,
        #         'drug_smiles': test_drugs[idx],
        #         'full_output': decoded_output,
        #         'extracted_answer': answer_text,
        #         'true_label': test_labels[idx]
        #     })
        #
        # if idx < 3:  # 显示前3个样本的解析结果
        #     print(f'Extracted answer: {answer_text}')
        #     print(f'Prediction (A->0, B->1): {pred}')
        #     print(f'True label: {test_labels[idx]}')

    # 版本1: 过滤掉None值
    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    filtered_predictions = [predictions[i] for i in valid_indices]
    filtered_labels = [test_labels[i] for i in valid_indices]

    # 版本2: None值设为0.5（表示完全不确定）
    predictions_with_uncertain = []
    for pred in predictions:
        if pred is None:
            predictions_with_uncertain.append(0.5)
        else:
            predictions_with_uncertain.append(float(pred))

    # 打印所有解析失败的样本
    if len(failed_parses) > 0:
        print('\n' + '=' * 80)
        print(f'FAILED PARSES (Total: {len(failed_parses)})')
        print('=' * 80)
        for fail in failed_parses:
            print(f'\n[Failed Sample {fail["index"]}]')
            print(f'Drug SMILES: {fail["drug_smiles"][:80]}...')
            print(f'True Label: {fail["true_label"]}')
            print(f'Extracted Answer: {fail["extracted_answer"]}')
            print(f'Full Output:\n{fail["full_output"]}')
            print('-' * 80)
    else:
        print('\nNo failed parses!')


    # 计算AUROC - 版本1 (过滤None)
    # —— 计算 AUROC（仅用有定义的 p_hat）——
    valid_idx = [i for i, p in enumerate(p_scores) if p is not None]
    y_true = [test_labels[i] for i in valid_idx]
    y_score = [p_scores[i] for i in valid_idx]

    print('y_true', y_true)
    print('y_score', y_score)

    print('\n' + '=' * 80)
    print('EVALUATION RESULTS (MC sampling, None dropped)')
    print('=' * 80)
    dropped = len(p_scores) - len(valid_idx)
    print(f'Items with defined p-hat: {len(valid_idx)}/{len(p_scores)} (dropped={dropped})')
    if len(valid_idx) > 0 and len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_score)
        print(f'AUROC: {auroc:.4f}')

        # 绘制ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {task_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # 保存图像
        plt.savefig(fig_save_path/f'roc_curve_{task_name}_{model_name.split("/")[-1]}.png', dpi=300, bbox_inches='tight')
        print(f'\nROC curve saved to: roc_curve_{task_name}.png')
        plt.close()
    else:
        print('Cannot compute AUROC (no valid items or only one class).')

    # 可选：汇总解析统计
    total_a = sum(a for a, b, n in per_item_stats)
    total_b = sum(b for a, b, n in per_item_stats)
    total_none = sum(n for a, b, n in per_item_stats)
    print(f'\nParse counts across all samples: A={total_a}, B={total_b}, None={total_none}')
    print('=' * 80)


if __name__ == "__main__":
    main()
