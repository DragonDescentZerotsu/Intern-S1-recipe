from unsloth import FastLanguageModel
import torch
import argparse
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer



def build_text(example):
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    text = f"User: {inp}\nAssistant: {out}"
    return {"text": text}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, default="zai-org/GLM-4.7-Flash")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--packing", action="store_true")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Unsloth 常用项
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA：更省显存（但看你的需求）")

    return p.parse_args()


def main():
    args = parse_args()
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/GLM-4.7-Flash",
        max_seq_length = 2048, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = True, # [NEW!] We have full finetuning now!
        trust_remote_code = True,
        unsloth_force_compile = False,
    )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",
#                       "out_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )