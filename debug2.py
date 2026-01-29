import os, multiprocessing as mp

# 1) 强制 vLLM 用 spawn，而不是默认的 fork
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"   # vLLM 官方建议
os.environ.setdefault("VLLM_USE_V1", "1")

# 2) 在 Python 侧把多进程 start method 切成 spawn（双保险）
mp.set_start_method("spawn", force=True)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'  # TODO: 根据需要调整 GPU

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json
from pathlib import Path
import argparse

# Configuration
MODEL_NAME = "Kiria-Nozan/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16-1-epoch-TDC-binary-wo-hergC_ToxCast_butkiewicz"
TEMPERATURE = 0.7
TOP_P = 1.0
MAX_TOKENS = 512
TENSOR_PARALLEL_SIZE = 4
GPU_MEMORY_UTILIZATION = 0.92
MAX_MODEL_LEN = 1024 * 24

# 每个 label 生成多少个样本
NUM_SAMPLES_PER_LABEL = 10


def to_prompt(text: str, tokenizer) -> str:
    """
    将用户文本经 chat template 展开，并在末尾追加 assistant 起始（生成位点）
    """
    messages = [{"role": "user", "content": text}]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.apply_chat_template(messages, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description='Debug script for testing vLLM model generation')
    parser.add_argument('--custom-prompt', '-p', type=str, default="How is today's weather?",  # None
                        help='Custom prompt text to generate from. If provided, only this prompt will be processed.')
    parser.add_argument('--num-samples', '-n', type=int, default=10,
                        help='Number of samples per label for batch testing (default: 10)')
    return parser.parse_args()


def run_custom_prompt(llm, tokenizer, sampling_params, custom_text):
    """处理用户自定义的 prompt"""
    print("=" * 80)
    print("CUSTOM PROMPT MODE")
    print("=" * 80)
    print("\nINPUT TEXT:")
    print("-" * 40)
    print(custom_text)
    print("-" * 40)
    
    # 构造 prompt
    prompt = to_prompt(custom_text, tokenizer)
    print("\nFORMATTED PROMPT (with chat template):")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    
    # 生成输出
    print("\nGenerating response...")
    outputs = llm.generate([prompt], sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    
    print("\n" + "=" * 80)
    print("MODEL OUTPUT:")
    print("=" * 80)
    print(generated_text)
    print("=" * 80)
    print("\nGeneration completed successfully!")


def run_batch_test(llm, tokenizer, sampling_params, num_samples_per_label):
    """批量测试 label=0 和 label=1 的样本"""
    # 读取测试数据
    test_data_path = Path(__file__).parent / "DataPrepare/TDC_test_prompts_label_scaffold/Skin_Reaction.jsonl"
    
    all_samples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))
    
    # 分离 label=0 和 label=1 的样本
    samples_label_0 = [s for s in all_samples if s['Y'] == 0]
    samples_label_1 = [s for s in all_samples if s['Y'] == 1]
    
    print(f"Total samples: {len(all_samples)}")
    print(f"Samples with label=0: {len(samples_label_0)}")
    print(f"Samples with label=1: {len(samples_label_1)}")
    
    # 选取每个 label 的前 N 个样本
    selected_0 = samples_label_0[:num_samples_per_label]
    selected_1 = samples_label_1[:num_samples_per_label]
    
    print(f"\nSelected {len(selected_0)} samples with label=0")
    print(f"Selected {len(selected_1)} samples with label=1")
    
    # 合并所有选中的样本
    all_selected = []
    for s in selected_0:
        all_selected.append((s, 0))
    for s in selected_1:
        all_selected.append((s, 1))
    
    # 构造所有 prompts
    prompts = []
    for sample, label in all_selected:
        prompt = to_prompt(sample['text'], tokenizer)
        prompts.append(prompt)
    
    print(f"\nGenerating responses for {len(prompts)} samples...")
    
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)
    
    # 显示结果
    print("\n" + "=" * 100)
    print("RESULTS FOR LABEL=0 (Ground Truth: Does NOT cause skin reaction)")
    print("=" * 100)
    
    correct_0 = 0
    for i, (sample, label) in enumerate(all_selected[:len(selected_0)]):
        generated_text = outputs[i].outputs[0].text.strip()
        # 简单判断预测是否正确
        pred_correct = "(A)" in generated_text and "(B)" not in generated_text
        if pred_correct:
            correct_0 += 1
        status = "✓" if pred_correct else "✗"
        
        print(f"\n[Sample {i+1}] {status}")
        print(f"  SMILES: {sample['text'].split('Drug SMILES:')[1].split('Please')[0].strip()[:80]}...")
        print(f"  Ground Truth: {label} (A: no reaction)")
        print(f"  Model Output: {generated_text[:100]}")
    
    print(f"\nAccuracy for label=0: {correct_0}/{len(selected_0)} = {correct_0/len(selected_0)*100:.1f}%")
    
    print("\n" + "=" * 100)
    print("RESULTS FOR LABEL=1 (Ground Truth: CAUSES skin reaction)")
    print("=" * 100)
    
    correct_1 = 0
    for i, (sample, label) in enumerate(all_selected[len(selected_0):]):
        idx = i + len(selected_0)
        generated_text = outputs[idx].outputs[0].text.strip()
        # 简单判断预测是否正确
        pred_correct = "(B)" in generated_text
        if pred_correct:
            correct_1 += 1
        status = "✓" if pred_correct else "✗"
        
        print(f"\n[Sample {i+1}] {status}")
        print(f"  SMILES: {sample['text'].split('Drug SMILES:')[1].split('Please')[0].strip()[:80]}...")
        print(f"  Ground Truth: {label} (B: causes reaction)")
        print(f"  Model Output: {generated_text[:100]}")
    
    print(f"\nAccuracy for label=1: {correct_1}/{len(selected_1)} = {correct_1/len(selected_1)*100:.1f}%")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    total_correct = correct_0 + correct_1
    total_samples = len(selected_0) + len(selected_1)
    print(f"Overall Accuracy: {total_correct}/{total_samples} = {total_correct/total_samples*100:.1f}%")
    print(f"Label=0 Accuracy: {correct_0}/{len(selected_0)} = {correct_0/len(selected_0)*100:.1f}%")
    print(f"Label=1 Accuracy: {correct_1}/{len(selected_1)} = {correct_1/len(selected_1)*100:.1f}%")
    print("=" * 100)
    print("\nGeneration completed successfully!")


def main():
    args = get_args()
    
    print(f"Loading model: {MODEL_NAME}")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 vLLM 模型
    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        # limit_mm_per_prompt={"video": 0, "image": 0},
    )

    # 设置生成参数
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    # 根据参数选择运行模式
    if args.custom_prompt:
        run_custom_prompt(llm, tokenizer, sampling_params, args.custom_prompt)
    else:
        run_batch_test(llm, tokenizer, sampling_params, args.num_samples)


if __name__ == "__main__":
    main()
