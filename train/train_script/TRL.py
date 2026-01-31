from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import torch
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import math
import os

def build_text(example):
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or "").strip()
    text = f"User: {inp}\nAssistant: {out}"
    return {"text": text}

def get_world_size():
    # 1) 如果 distributed 已初始化，这是最准的
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()

    # 2) torchrun / deepspeed / accelerate 通常会设置 WORLD_SIZE
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    # 3) Slurm 常用 NTASKS = 总进程数
    if "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])

    # 4) 最后兜底：单机单进程时用本地可见 GPU 数（注意：这不是跨节点总数）
    if torch.cuda.is_available():
        return torch.cuda.device_count()

    return 1

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name_or_path", type=str, default="zai-org/GLM-4.7-Flash")
    p.add_argument("--train_file", type=str, default='DataPrepare/SFT_data/SFT_data/GPT/GLM-4.7-Flash/TDC_SFT_data_binary_sm_wo_herg-c_ToxCast_butkiewicz/training.jsonl')
    p.add_argument("--eval_file", type=str, default='DataPrepare/SFT_data/SFT_data/GPT/GLM-4.7-Flash/TDC_SFT_data_binary_sm_wo_herg-c_ToxCast_butkiewicz/test_debug.jsonl')

    p.add_argument("--output_dir", type=str, default='checkpoints/GLM-4.7-Flash/TDC_binary_sm_wo_herg-c_ToxCast_butkiewica')
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--packing", action="store_true", default=True, help="Enable sequence packing")

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--num_train_epochs", type=float, default=8.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Unsloth 常用项
    p.add_argument("--load_in_4bit", action="store_true", help="QLoRA：更省显存（但看你的需求）")

    # WandB
    p.add_argument("--wandb_project", type=str, default="GLM-4.7-Flash-SFT")
    p.add_argument("--wandb_run_name", type=str, default=None)

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default="train/train_script/ds_zero3_config.json",
                   help="Path to DeepSpeed config file for ZeRO-3 model sharding")

    return p.parse_args()

def map_prompts_keys(examples):
   return { "prompt" : examples['input'], "completion": examples['output'], }

def formatting_func(examples):
    prompts = examples.get("prompt", examples.get("input", ""))
    completions = examples.get("completion", examples.get("output", ""))

    # 兼容：既可能是单条(str)，也可能是batch(list)
    if isinstance(prompts, str):
        prompts = [prompts]
        completions = [completions]

    texts = []
    for p, c in zip(prompts, completions):
        p = (p or "").strip()
        c = (c or "").strip()
        texts.append(f"{p}{c}")   # 你想用 + 就先用着
    return texts

def debug_loss_mask(trainer, tokenizer, idx=0, max_print=300):
    sample = trainer.train_dataset[idx]
    input_ids = sample["input_ids"]
    labels    = sample["labels"]

    # 1) 统计一下到底有多少 token 在训练
    n_total = len(labels)
    n_train = sum(l != -100 for l in labels)
    print(f"[idx={idx}] train_tokens = {n_train}/{n_total} ({n_train/n_total:.2%})")

    # 2) 把“参与loss的token”单独 decode 出来看
    visible_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    if len(visible_ids) == 0:
        print("WARNING: no visible training tokens (all labels are -100) -> loss will be ~0")
    else:
        print("=== Decoded tokens that contribute to loss (truncated) ===")
        print(tokenizer.decode(visible_ids[:max_print]))
        print("=== END ===")

    # 3) 更细：逐 token 打印（方便你肉眼确认边界）
    toks = tokenizer.convert_ids_to_tokens(input_ids[:max_print])
    for i, (tok, lab) in enumerate(zip(toks, labels[:max_print])):
        flag = "TRAIN" if lab != -100 else "MASK "
        print(f"{i:04d} {flag} {tok}")

# 用法：
# debug_loss_mask(trainer, tokenizer, idx=0)


def main():
    args = parse_args()

    world_size = get_world_size()
    print(f"World size: {world_size}")
    # exit(0)

    ds = load_dataset("json", data_files={"train": args.train_file, "eval": args.eval_file})
    ds = ds.map(map_prompts_keys)

    train_ds = ds["train"]
    eval_ds = ds["eval"]

    # Calculate eval_steps for 0.5 epoch intervals
    # Total steps per epoch = num_train_samples / (per_device_batch_size * num_gpus * gradient_accumulation_steps)
    # We use eval_strategy="steps" but calculate steps to match epoch fractions
    num_train_samples = len(train_ds)
    # Assume world_size will be set by accelerate/torchrun
    # For now, calculate based on single device, will be adjusted at runtime
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/GLM-4.7-Flash",
        max_seq_length = args.max_seq_length, # Choose any for long context!
        load_in_4bit = False,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = True, # [NEW!] We have full finetuning now!
        trust_remote_code = True,
        unsloth_force_compile = False,
        device_map = None,
    )

    # SFTConfig replaces TrainingArguments for TRL SFTTrainer
    # Multi-GPU training is supported via accelerate/torchrun (run with: accelerate launch or torchrun)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        optim="paged_adamw_8bit",
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        # Evaluation strategy: by epoch with fractional epochs via steps
        eval_strategy="steps",
        # eval_steps=eval_steps,  # Approximately 0.5 epoch
        save_strategy="steps",
        # save_steps=eval_steps,  # Save at same frequency as eval
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        # WandB for experiment tracking
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
        # Sequence packing settings
        packing=args.packing,
        max_seq_length=args.max_seq_length,
        # DeepSpeed ZeRO-3 for model sharding across GPUs
        deepspeed=args.deepspeed,
        # Gradient checkpointing to reduce memory usage
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Load best model at end
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=False
    )

    # TRL SFTTrainer with SFTConfig
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,  # Required by Unsloth
    )

    num_train_samples_after_packing = len(trainer.get_train_dataloader())

    steps_per_epoch_estimate = num_train_samples_after_packing // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    eval_steps = max(1, steps_per_epoch_estimate // 2)  # 0.5 epoch

    trainer.args.eval_steps = 5 # eval_steps
    trainer.args.save_steps = 5 # eval_steps

    # print(f"Average pack length: {num_train_samples / num_train_samples_after_packing}")
    print(f"Num train samples after packing: {num_train_samples_after_packing}")
    print(f'One epoch steps: {steps_per_epoch_estimate}')
    print(f"Eval steps: {eval_steps}")

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "[gMASK]<sop><|user|>", # Updated for GLM
        response_part = "<|assistant|></think>",
    )

    print('='*80)
    print('Training data example:')
    print('input_ids idx-100:')
    print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))
    print('-'*80)
    # print('labels:')
    # print(tokenizer.decode(trainer.train_dataset[100]["labels"]))
    # print('-'*80)
    print('input_ids idx-1:')
    print(tokenizer.decode(trainer.train_dataset[1]["input_ids"]))
    print('-'*80)
    # print('labels:')
    # print(tokenizer.decode(trainer.train_dataset[1]["labels"]))
    print('='*80)

    # exit(0)

    debug_loss_mask(trainer, tokenizer, idx=0)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done. Saved to:", args.output_dir)


if __name__ == "__main__":
    import os
    # Set WandB project name via environment variable
    os.environ.setdefault("WANDB_PROJECT", "GLM-4.7-Flash-SFT")
    main()
