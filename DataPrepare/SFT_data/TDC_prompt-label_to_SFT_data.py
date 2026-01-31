
"""
This script prepares SFT (Supervised Fine-Tuning) data for TDC (Therapeutics Data Commons) tasks.

It processes JSONL files from a specific input directory (TDC_test_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz),
converting binary labels into a multiple-choice format ((A) vs (B)).
The script standardizes the prompts by removing specific Chain-of-Thought (CoT) suffixes.
It supports outputting data in either Alpaca-style JSONL format (instruction, input, output) 
or GPT-style JSONL format (input, output) using a chat template, configurable via DATA_STYLE.


该脚本用于为 TDC(Therapeutics Data Commons)任务 准备 SFT(监督微调,Supervised Fine-Tuning) 数据。

它会从指定的输入目录
TDC_test_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz
中读取并处理 JSONL 文件,将原本的二分类标签转换为多项选择格式((A) vs (B))。

此外,该脚本还会通过移除特定的 Chain-of-Thought(CoT)后缀 来统一(标准化)prompt 的格式。
支持通过 DATA_STYLE 配置将数据输出为 Alpaca 风格的 JSONL 格式(包含 instruction、input、output 字段)
或经过 Chat Template 处理的 GPT 风格 JSONL 格式(包含 input、output 字段),以便用于不同模型的训练。
"""

import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def process_dataset(input_dir, output_file, data_style, model_name, tokenizer, enable_thinking):
    """
    Process a single dataset (train/test/valid) and write to output file.
    
    Args:
        input_dir: Path to input directory containing JSONL files
        output_file: Path to output JSONL file
        data_style: 'GPT' or 'Alpaca'
        model_name: Model name for GPT style processing
        tokenizer: Tokenizer for GPT style processing (can be None for Alpaca)
        enable_thinking: Whether to enable thinking for chat template
    
    Returns:
        Number of samples processed
    """
    print(f"\nProcessing Dataset:")
    print(f"  Input Directory: {input_dir}")
    print(f"  Output File: {output_file}")
    
    if not input_dir.exists():
        print(f"  Warning: Input directory {input_dir} does not exist. Skipping.")
        return 0

    # Get list of all jsonl files
    jsonl_files = list(input_dir.glob('*.jsonl'))
    print(f"  Found {len(jsonl_files)} JSONL files to process.")
    
    if len(jsonl_files) == 0:
        print(f"  No JSONL files found. Skipping.")
        return 0
    
    total_samples = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in tqdm(jsonl_files, desc=f"Processing {output_file.name}"):
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        
                        # Replace the long prompt suffix with 'Answer:'
                        cot_suffix = 'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
                        text = text.replace(cot_suffix, 'Answer:')
                        
                        y_label = data.get('Y')
                        
                        if y_label == 1:
                            output_val = "(B)"
                        elif y_label == 0:
                            output_val = "(A)"
                        else:
                            print(f"Warning: Unexpected label {y_label} in file {input_file.name}. Skipping line.")
                            continue
                            
                        if data_style == 'Alpaca':
                            data_entry = {
                                "instruction": text,
                                "input": "",
                                "output": output_val
                            }
                        elif data_style == 'GPT':
                            message = [
                                {"role": "user", "content": text},
                                {"role": "assistant", "content": output_val}
                            ]
                            if model_name == "internlm/Intern-S1-mini":
                                rendered = tokenizer.apply_chat_template(
                                    message,
                                    tokenize=False,
                                    add_generation_prompt=False,
                                    enable_thinking=enable_thinking
                                )
                                input_text = rendered.split("assistant")[0] + "assistant"
                                answer_text = rendered.split("assistant")[1]
                            elif model_name == "openai/gpt-oss-20b":
                                rendered = tokenizer.apply_chat_template(
                                    message,
                                    tokenize=False,
                                    add_generation_prompt=False,
                                    # enable_thinking=enable_thinking
                                )
                                split_mark = 'assistant<|channel|>final<|message|>'
                                input_text = rendered.split(split_mark)[0] + split_mark
                                answer_text = rendered.split(split_mark)[1]
                            elif model_name == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16":
                                rendered = tokenizer.apply_chat_template(
                                    message,
                                    tokenize=False,
                                    add_generation_prompt=False,
                                    enable_thinking=enable_thinking
                                )
                                split_mark = 'assistant\n<think></think>'
                                input_text = rendered.split(split_mark)[0] + split_mark
                                answer_text = rendered.split(split_mark)[1]
                            elif model_name == "zai-org/GLM-4.7-Flash":
                                rendered = tokenizer.apply_chat_template(
                                    message,
                                    tokenize=False,
                                    add_generation_prompt=False,
                                    enable_thinking=enable_thinking
                                )
                                split_mark = '<|assistant|></think>'
                                input_text = rendered.split(split_mark)[0] + split_mark
                                answer_text = rendered.split(split_mark)[1]
                            data_entry = {"input": input_text, "output": answer_text}
                        
                        f_out.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                        total_samples += 1
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to decode JSON in file {input_file.name}. Skipping line.")
                    except Exception as e:
                        print(f"Error processing line in {input_file.name}: {e}")

    print(f"  Completed: {total_samples} samples saved to {output_file}")
    return total_samples


def main():
    # Define directories
    current_dir = Path(__file__).parent.resolve() # DataPrepare/SFT_data
    project_root = current_dir.parent.parent # /data1/tianang/Projects/Intern-S1/
    
    # Configuration
    DATA_STYLE = 'GPT'  # 'GPT' or 'Alpaca'
    model_name = "zai-org/GLM-4.7-Flash"
                 # "openai/gpt-oss-20b"
                 # "internlm/Intern-S1-mini"
                 # "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
                 # zai-org/GLM-4.7-Flash
    ENABLE_THINKING = False

    # Define dataset configurations for GPT style
    # Maps: (input_dir_suffix, output_filename)
    # input_dir_suffix: train, test, valid
    # output_filename: training.jsonl, test.jsonl, validation.jsonl
    gpt_dataset_configs = [
        ('train', 'training.jsonl'),
        ('test', 'test.jsonl'),
        # ('valid', 'validation.jsonl'),
    ]

    tokenizer = None

    # Output details
    if DATA_STYLE == 'Alpaca':
        # Alpaca style is for llamafactory use, all the models should be the same 
        # output_subdir = f'Alpaca/{model_name.split("/")[-1]}/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz'
        output_subdir = f'Alpaca/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz'
        output_filename = 'TDC_SFT_data_scaffold_wo_herg-c_ToxCast_butkiewicz.jsonl'
        
        input_dir = project_root / 'DataPrepare' / 'TDC_train_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz'
        output_dir = current_dir / 'SFT_data' / output_subdir
        output_file = output_dir / output_filename
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process single dataset for Alpaca style
        total_samples = process_dataset(
            input_dir, output_file, DATA_STYLE, model_name, tokenizer, ENABLE_THINKING
        )
        print(f"\n{'='*60}")
        print(f"Processing complete. Total {total_samples} samples saved.")
        
    elif DATA_STYLE == 'GPT':
        output_subdir = f'GPT/{model_name.split("/")[-1]}/TDC_SFT_data_binary_sm_wo_herg-c_ToxCast_butkiewicz'
        
        # Load tokenizer for GPT format
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        output_dir = current_dir / 'SFT_data' / output_subdir
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output Directory: {output_dir}")
        print(f"Processing {len(gpt_dataset_configs)} datasets (train/test/valid)...")
        
        grand_total = 0
        for split_name, output_filename in gpt_dataset_configs:
            # Construct input directory path
            # train -> TDC_train_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz
            # test -> TDC_test_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz
            # valid -> TDC_valid_prompts_label_scaffold_wo_herg-c_ToxCast_butkiewicz
            input_dir = project_root / 'DataPrepare' / f'TDC_{split_name}_prompts_label_sm_wo_herg-c_ToxCast_butkiewicz'
            output_file = output_dir / output_filename
            
            samples = process_dataset(
                input_dir, output_file, DATA_STYLE, model_name, tokenizer, ENABLE_THINKING
            )
            grand_total += samples
        
        print(f"\n{'='*60}")
        print(f"All processing complete. Grand total: {grand_total} samples saved.")

if __name__ == "__main__":
    main()
