# train_fp8_te_sft.py
import os
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import TERecipeKwargs, get_active_deepspeed_plugin  # ✅ 推荐：显式 TE recipe handler
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import deepspeed
import copy

current_dir = Path(__file__).parent.resolve()

MODEL_NAME = "internlm/Intern-S1"
TOK_PATH   = current_dir.parent / "SFT_data" / "SFT_data" / "TDC_SFT_data_all_tasks.arrow"
OUTPUT_DIR = current_dir.parent / "checkpoints" / MODEL_NAME.split('/')[-1] / "lora" / "sft-fp8"

BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-5
EPOCHS = 1
WARMUP_RATIO = 0.1
LOG_EVERY = 20
PAD_TO_MULTIPLE = 16  # ✅ 为 Tensor Core/FP8 友好
ZERO_STAGE = 3

@dataclass
class Collator:
    pad_id: int
    pad_to_multiple: int = None

    def __call__(self, batch):
        input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
        labels    = [torch.tensor(x["labels"],    dtype=torch.long) for x in batch]

        # 计算本批最长长度，并向上取整到 pad_to_multiple
        max_len = max(seq.size(0) for seq in input_ids)
        if self.pad_to_multiple is not None and self.pad_to_multiple > 1:
            max_len = int(math.ceil(max_len / self.pad_to_multiple) * self.pad_to_multiple)

        def _pad_to(x, length, pad_val):
            if x.size(0) < length:
                return torch.cat([x, x.new_full((length - x.size(0),), pad_val)])
            return x

        input_ids = [ _pad_to(x, max_len, self.pad_id) for x in input_ids ]
        labels    = [ _pad_to(x, max_len, -100)        for x in labels    ]

        input_ids = torch.stack(input_ids, dim=0)
        labels    = torch.stack(labels,    dim=0)
        attention_mask = (input_ids != self.pad_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

def main():
    # ✅ 推荐：在 Accelerator 中直接启用 FP8 + TE 配方
    te_kwargs = TERecipeKwargs(
        fp8_format="HYBRID",        # 与官方示例一致
        amax_history_len=32,
        amax_compute_algo="max",
        use_autocast_during_eval=False,  # 评估期默认禁用 FP8，更稳
    )
    # accelerator = Accelerator(
    #     mixed_precision="fp8",
    #     kwargs_handlers=[te_kwargs],
    #     gradient_accumulation_steps=GRAD_ACCUM,  # 可选：让框架来管累积步
    # )
    ds_plugin = DeepSpeedPlugin(
            zero_stage = ZERO_STAGE,
        zero3_init_flag = (ZERO_STAGE == 3),  # ✅ Zero-3 初始化加速（官方示例同款）
    )
    accelerator = Accelerator(
        mixed_precision = "fp8",
        kwargs_handlers = [te_kwargs],
        gradient_accumulation_steps = GRAD_ACCUM,  # ✅ 累积步交给 Accelerate/DeepSpeed
        deepspeed_plugin = ds_plugin,  # ✅ 启用 DeepSpeed
    )
  # ✅ 设置 per-GPU 的 micro-batch（DeepSpeed 配置项）
    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = BATCH_SIZE
  # ✅ 如需在 ZeRO-3 下 save_pretrained 聚合权重：

    if ZERO_STAGE == 3:
        accelerator.state.deepspeed_plugin.deepspeed_config.setdefault(
                "zero_optimization", {}
                                      ).setdefault(
                "stage3_gather_16bit_weights_on_model_save", True
                                                              )
    device = accelerator.device

    # 小优化
    torch.backends.cuda.matmul.allow_tf32 = True

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    accelerator.print("Loading dataset...")
    ds = load_from_disk(str(TOK_PATH))
    accelerator.print("Dataset loaded.")

    # ds = ds.map(lambda x: {"length": len(x["input_ids"])}, desc="add length", num_proc=32)
    #
    # max_len = max(ds["length"])
    # min_len = min(ds["length"])
    # print("max_len:", max_len, "min_len:", min_len)
    #
    # # 按长度从大到小排序
    # ds_sorted = ds.sort("length", reverse=True)

    collator = Collator(pad_id=tokenizer.pad_token_id, pad_to_multiple=PAD_TO_MULTIPLE)
    dl = DataLoader(
        ds,  # TODO: ds_sorted
        batch_size=BATCH_SIZE,
        shuffle=True,  # TODO
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # model（权重用 bf16；计算由 Accelerate 以 FP8/TE 管理）[

    if ZERO_STAGE == 3:
        # ds_plugin = get_active_deepspeed_plugin(accelerator.state)
        # ds_cfg = copy.deepcopy(ds_plugin.deepspeed_config)

        # 2) 确保 batch 相关是整数（不要是 "auto"）
        # ds_cfg["train_micro_batch_size_per_gpu"] = int(BATCH_SIZE)
        # ds_cfg["gradient_accumulation_steps"] = int(GRAD_ACCUM)
        with deepspeed.zero.Init():  # ★ 官方推荐写法
            accelerator.print('Preparing base model in zero stage 3...')
            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,  # 权重 dtype
                trust_remote_code=True,
                low_cpu_mem_usage=True,  # ★ 有助于进一步降低主机内存峰值
            )
            base_model.config.use_cache = False
            base_model.gradient_checkpointing_enable()

            # LoRA 需在 prepare() 之前包裹
            accelerator.print('Preparing LoRA...')
            lora_cfg = LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            model = get_peft_model(base_model, lora_cfg)
            accelerator.print('LoRA prepared.')
    else:
        # === base model (bf16 weights; compute由Accelerate以FP8管理) ===
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()  # 与 LoRA 兼容

        # === LoRA：仅此处为新增/改动 ===
        accelerator.print('Preparing LoRA...')
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base_model, lora_cfg)
        accelerator.print('LoRA prepared.')

    # 先 prepare，再据此计算 scheduler 的总步数
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)

    # 让 Accelerate 接管模型/优化器/数据
    accelerator.print('Preparing accelerator model...')
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)
    accelerator.print('Accelerator model prepared.')

    # ✅ 准确计算总更新步数（分布式场景下 len(dl) 已变化）
    num_update_steps_per_epoch = math.ceil(len(dl) / 1)  # 因为我们把累积步交给 Accelerator 了
    total_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    # scheduler 也交给 Accelerate（不强制，但保持一致）
    lr_scheduler = accelerator.prepare(lr_scheduler)

    model.train()
    pbar = tqdm(total=total_steps, disable=not accelerator.is_local_main_process)
    step = 0

    for epoch in range(EPOCHS):
        for batch in dl:
            # ✅ 不再手动套 fp8/bf16 autocast；Accelerate 已处理
            with accelerator.accumulate(model):
                outputs = model(**{k: v.to(device, non_blocking=True) for k, v in batch.items()})
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % LOG_EVERY == 0:
                accelerator.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1
            pbar.update(1)
    pbar.close()

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        accelerator.print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
