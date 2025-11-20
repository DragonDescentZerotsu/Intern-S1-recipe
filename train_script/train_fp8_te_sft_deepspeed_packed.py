# train_fp8_te_sft.py
import os
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import TERecipeKwargs  # ✅ 推荐：显式 TE recipe handler
from tqdm import tqdm
from torchtune.datasets import PackedDataset

current_dir = Path(__file__).parent.resolve()

MODEL_NAME = "internlm/Intern-S1-mini"
TOK_PATH   = current_dir.parent / "SFT_data" / "SFT_data" / "TDC_SFT_data_all_tasks.arrow"
PACKED_TOK_PATH = current_dir.parent / "SFT_data" / "SFT_data" / "TDC_SFT_data_all_tasks_packed.arrow"
OUTPUT_DIR = current_dir.parent / "checkpoints" / "Intern-S1-mini" / "full" / "sft-fp8"

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
    max_length = tokenizer.model_max_length
    pad_token_id = tokenizer.pad_token_id

    # dataset
    if PACKED_TOK_PATH.exists():
        # 如果已有缓存的 packed dataset，直接加载
        accelerator.print(f"Loading cached packed dataset from {PACKED_TOK_PATH}...")
        pack_ds = load_from_disk(str(PACKED_TOK_PATH))
        accelerator.print("Packed dataset loaded from cache.")
    else:
        # 否则，加载原始数据集并进行 packing
        accelerator.print(f"Loading original dataset from {TOK_PATH}...")
        ds = load_from_disk(str(TOK_PATH))
        accelerator.print("Dataset loaded.")

        accelerator.print("Creating packed dataset...")
        ds = ds.rename_column("input_ids", "tokens")
        pack_ds = PackedDataset(ds, max_seq_len=max_length, padding_idx=pad_token_id, split_across_pack=False)

        # 将 PackedDataset 转换为 HF Dataset 并保存
        if accelerator.is_main_process:
            accelerator.print(f"Saving packed dataset to {PACKED_TOK_PATH}...")
            from datasets import Dataset as HFDataset

            # 提取所有打包后的数据
            packed_data = {
                "tokens": [],
                "labels": [],
                "input_pos": [],
                "seq_lens": []
            }

            for i in range(len(pack_ds)):
                item = pack_ds[i]
                packed_data["tokens"].append(item["tokens"].tolist())
                packed_data["labels"].append(item["labels"].tolist())
                packed_data["input_pos"].append(item["input_pos"].tolist())
                packed_data["seq_lens"].append(item["seq_lens"].tolist())

            # 创建 HF Dataset 并保存
            hf_packed = HFDataset.from_dict(packed_data)
            hf_packed.save_to_disk(str(PACKED_TOK_PATH))
            accelerator.print(f"Packed dataset saved to {PACKED_TOK_PATH}")

        # 等待主进程保存完成
        accelerator.wait_for_everyone()

        # 所有进程重新加载保存的 packed dataset，确保一致性
        accelerator.print(f"Reloading packed dataset from {PACKED_TOK_PATH}...")
        pack_ds = load_from_disk(str(PACKED_TOK_PATH))
        accelerator.print("Packed dataset reloaded.")

    class TTWrapped(Dataset):
        def __init__(self, ttd): self.ttd = ttd

        def __len__(self): return len(self.ttd)

        def __getitem__(self, i):
            ex = self.ttd[i]
            return {
                "input_ids": torch.tensor(ex["tokens"], dtype=torch.long),
                "labels": torch.tensor(ex["labels"], dtype=torch.long),
                "position_ids": torch.tensor(ex["input_pos"], dtype=torch.long),
                "seq_lens": torch.tensor(ex["seq_lens"], dtype=torch.int32),  # ← 用它构造块掩码
            }

    packed_ds = TTWrapped(pack_ds)

    collator = Collator(pad_id=tokenizer.pad_token_id, pad_to_multiple=PAD_TO_MULTIPLE)
    dl = DataLoader(
        packed_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # collate_fn=collator,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # model（权重用 bf16；计算由 Accelerate 以 FP8/TE 管理）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # 训练期关闭
    model.gradient_checkpointing_enable()

    # 先 prepare，再据此计算 scheduler 的总步数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)

    # 让 Accelerate 接管模型/优化器/数据
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

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

    def make_block_causal_mask_additive(batch_seq_lens, max_len, device, dtype):
        B = len(batch_seq_lens)
        # 负无穷用该 dtype 的最小值（Molmo 也这么做），避免 autocast 下精度问题
        NEG_INF = torch.finfo(dtype).min
        mask = torch.full((B, 1, max_len, max_len), fill_value=NEG_INF, device=device, dtype=dtype)
        for b, lens in enumerate(batch_seq_lens):
            start = 0
            total = int(torch.sum(lens).item())
            for Lk in lens.tolist():
                s, e = start, start + Lk
                tri = torch.tril(torch.ones((Lk, Lk), dtype=torch.bool, device=device))
                block = mask[b, 0, s:e, s:e]
                block[tri] = 0  # 允许处=0（参与注意力）
                start = e
        return mask

    for epoch in range(EPOCHS):
        for batch in dl:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pos_ids = batch["position_ids"].to(device, non_blocking=True)
            lens_list = [x.to(device) for x in batch["seq_lens"]]  # List[1D Tensor]

            L = input_ids.size(1)  # = PACKED_LEN
            compute_dtype = next(model.parameters()).dtype  # 在 TE FP8 下通常是 torch.bfloat16
            attn_mask = make_block_causal_mask_additive(lens_list, L, device, compute_dtype)
            # ✅ 不再手动套 fp8/bf16 autocast；Accelerate 已处理
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=input_ids,
                    labels=labels,
                    position_ids=pos_ids,  # torchtune 给的是“段内相对位置”，model一般可直接用
                    attention_mask=attn_mask,  # 4D 加性掩码
                )
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
