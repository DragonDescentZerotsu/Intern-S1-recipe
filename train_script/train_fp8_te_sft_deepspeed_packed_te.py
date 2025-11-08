# train_te_fp8_packed.py  ——  B-2: Transformer Engine FP8 Attention + Packed Dataset (varlen)
import os, math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import TERecipeKwargs, contextual_fp8_autocast  # ★
from transformer_engine.common.recipe import DelayedScaling, Format   # ★

# ★ 新：TE 注意力
from transformer_engine.pytorch import DotProductAttention  # TE FP8/cudnn/flash 多后端统一API

current_dir = Path(__file__).parent.resolve()

MODEL_NAME = "internlm/Intern-S1-mini"
TOK_PATH   = current_dir.parent / "SFT_data" / "SFT_data" / "TDC_SFT_data_all_tasks.arrow"
OUTPUT_DIR = current_dir.parent / "checkpoints" / "Intern-S1-mini" / "full" / "sft-fp8-te-attn-varlen"

BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-5
EPOCHS = 1
WARMUP_RATIO = 0.1
LOG_EVERY = 20
PACK_TO = 4096          # ★ 目标打包长度（可做 1k/2k/4k/8k 网格搜索）
ZERO_STAGE = 3

# -------------------------
# 1) 数据：packing + cu_seqlens
# -------------------------
@dataclass
class TEPackingCollator:
    pad_id: int
    pack_to: int = 4096       # 建议本身也设成 8 的倍数

    def __call__(self, batch):
        cu = [0]; cur = 0; ids_all=[]; labs_all=[]
        for ex in batch:
            ids = torch.tensor(ex["input_ids"], dtype=torch.long)
            lab = torch.tensor(ex["labels"],    dtype=torch.long)
            L = ids.numel()
            if cur>0 and cur+L>self.pack_to:
                break   # 本 batch 只打一个包，更稳
            ids_all.append(ids); labs_all.append(lab)
            cur += L; cu.append(cur)

        input_ids = torch.cat(ids_all,  dim=0)
        labels    = torch.cat(labs_all, dim=0)

        # ---- 关键：为 FP8 Linear 对齐 (B*T) % 8 == 0，这里 B=1 => T % 8 == 0
        T = input_ids.numel()
        Lpad = (-T) % 8
        if Lpad:
            pad_ids   = torch.full((Lpad,), self.pad_id, dtype=torch.long)
            pad_labs  = torch.full((Lpad,), -100,        dtype=torch.long)
            input_ids = torch.cat([input_ids, pad_ids], dim=0)
            labels    = torch.cat([labels,    pad_labs],dim=0)
            cu.append(cu[-1] + Lpad)  # ★ 把对齐 padding 作为“独立末段”
        # -------------------------------------------------------------

        cu_seqlens = torch.tensor(cu, dtype=torch.int32)
        max_seqlen = torch.tensor(int((cu_seqlens[1:] - cu_seqlens[:-1]).max()), dtype=torch.int32)

        # HF 兼容的 mask（注意真正的 varlen 是看 cu_seqlens，而不是这个 mask）
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            "input_ids": input_ids.unsqueeze(0),   # [1, T_total(对齐)]
            "labels":    labels.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
            "cu_seqlens": cu_seqlens,             # 段前缀和（含末段对齐 padding）
            "max_seqlen": max_seqlen,
        }


# -------------------------
# 2) 模型：用 TE 注意力替换原自注意力
# -------------------------
class TEPackedSelfAttention(nn.Module):
    """
    轻量包装：复用原 Attention 的 q/k/v/o 投影，调用 TE.DotProductAttention。
    读取根模型的 model._te_packing_ctx 获取 cu_seqlens / max_seqlen。
    适配 LLaMA/Mistral/Intern 等常见命名：q_proj/k_proj/v_proj/o_proj。
    """
    def __init__(self, orig_attn: nn.Module, n_heads: int, head_dim: int, causal: bool = True):
        super().__init__()
        self.orig = orig_attn
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj
        self.num_heads = n_heads
        self.head_dim  = head_dim
        self.causal = causal
        # TE 统一 API
        self.te_attn = DotProductAttention()

        # 获取根模块引用（用于读取 _te_packing_ctx）
        # 假定模块注册后，会被 setattr 增加 _te_root 引用
        self._te_root = None

    def _shape_qkv(self, x: torch.Tensor):
        # x: [B, T, H*D] -> [T_total, H, D] （采用 THD varlen 格式，B=1 更自然）
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        x = x.view(B, T, H, D)
        # pack 后我们把 batch 已在数据侧合一：B 应为 1
        x = x.reshape(T, H, D).contiguous()
        return x

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        assert self._te_root is not None, "TEPackedSelfAttention: _te_root 未设置"
        ctx = getattr(self._te_root, "_te_packing_ctx", None)
        assert ctx is not None, "未在前向前设置 _te_packing_ctx（需要把 cu_seqlens/max_seqlen 传给注意力）"

        cu_seqlens = ctx["cu_seqlens"]         # int32 [N_seq+1]
        max_seqlen = int(ctx["max_seqlen"])    # Python int
        # 1) 线性投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # 2) reshape 到 TE 支持的 varlen THD
        q = self._shape_qkv(q)   # [T_total, H, D]
        k = self._shape_qkv(k)
        v = self._shape_qkv(v)

        # 3) TE 注意力（使用 padding_causal，传入 cu_seqlens）
        out = self.te_attn(
            q, k, v,
            # ★ 关键：mask 与 varlen
            mask_type="padding_causal" if self.causal else "padding",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            # 其他常用参数走默认：softmax_scale=None（自动 1/sqrt(d)），dropout 由外层训练态控制
        )   # 输出形状与 q 相同：[T_total, H, D]

        # 4) 还原回 [B, T, H*D]
        # 由于我们把 batch 合一，这里 B=1
        T_total = out.size(0)
        out = out.reshape(1, T_total, self.num_heads * self.head_dim).contiguous()
        out = self.o_proj(out)  # [1, T_total, C]
        return out

# ---------- utils: KV 头重复，等价于 Qwen3 repeat_kv ----------
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # x: [T, H_kv, D] -> [T, H_kv*n_rep, D]
    if n_rep == 1:
        return x
    T, Hkv, D = x.shape
    x = x.unsqueeze(2).expand(T, Hkv, n_rep, D).reshape(T, Hkv * n_rep, D)
    return x

class TEPackedSelfAttention(nn.Module):
    def __init__(self, orig_attn: nn.Module, n_heads: int, head_dim: int,
                 n_kv_heads: int = None, causal: bool = True):
        super().__init__()
        self.orig = orig_attn
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj
        self.num_heads = n_heads
        self.num_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim  = head_dim
        self.causal = causal
        self.te_attn = DotProductAttention()
        self._te_root = None

    def _shape_q(self, x: torch.Tensor):
        # [B, T, H*D] -> [T, H, D]
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim
        return x.view(B, T, H, D).reshape(T, H, D).contiguous()

    def _shape_kv(self, x: torch.Tensor):
        # [B, T, H_kv*D] -> [T, H_kv, D]
        B, T, C = x.shape
        Hkv, D = self.num_kv_heads, self.head_dim
        return x.view(B, T, Hkv, D).reshape(T, Hkv, D).contiguous()

    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        assert self._te_root is not None, "TEPackedSelfAttention: _te_root 未设置"
        ctx = getattr(self._te_root, "_te_packing_ctx", None)
        assert ctx is not None, "未设置 _te_packing_ctx（需要 cu_seqlens/max_seqlen）"

        cu_seqlens = ctx["cu_seqlens"]
        max_seqlen = int(ctx["max_seqlen"])

        # 线性投影
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 变形到 varlen (THD)
        q = self._shape_q(q)          # [T, H,   D]
        k = self._shape_kv(k)         # [T, Hkv, D]
        v = self._shape_kv(v)         # [T, Hkv, D]

        # 若是 GQA：把 KV 重复到与 Q 头数一致（等价 Qwen3 repeat_kv）
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = _repeat_kv(k, rep)    # [T, H, D]
            v = _repeat_kv(v, rep)    # [T, H, D]

        # TE 注意力（padding_causal + cu_seqlens varlen）
        out = self.te_attn(
            q, k, v,
            mask_type="padding_causal" if self.causal else "padding",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )  # [T, H, D]

        # 还原到 [B=1, T, H*D]
        T_total = out.size(0)
        out = out.reshape(1, T_total, self.num_heads * self.head_dim).contiguous()
        out = self.o_proj(out)
        return out

def _swap_attn_with_te(model: nn.Module):
    # 1) 找层容器（Intern-S1/Qwen3 常见：model.model.layers 或 model.language_model.layers）
    layers = None
    for attr in ("model", "language_model"):
        if hasattr(model, attr):
            maybe = getattr(model, attr)
            if hasattr(maybe, "layers"):
                layers = maybe.layers
                break
    if layers is None:
        return 0

    # 2) 兜底配置（来自 config，比直接查 weight 可靠）
    cfg = getattr(model, "config", None)
    cfg_hidden = getattr(cfg, "hidden_size", None)
    cfg_heads  = getattr(cfg, "num_attention_heads", None)
    cfg_kv     = getattr(cfg, "num_key_value_heads", None) or cfg_heads

    swapped = 0
    for layer in layers:
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        if not all(hasattr(attn, x) for x in ("q_proj", "k_proj", "v_proj", "o_proj")):
            continue

        # 优先用模块属性，其次 config；都没有就报错
        num_heads = getattr(attn, "num_heads", None) \
                    or getattr(attn, "n_heads", None) \
                    or getattr(attn, "num_attention_heads", None) \
                    or cfg_heads
        num_kv = getattr(attn, "num_key_value_heads", None) or cfg_kv
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            assert cfg_hidden is not None and num_heads is not None, "无法从 config 推断 head_dim"
            head_dim = cfg_hidden // int(num_heads)

        te_mod = TEPackedSelfAttention(
            attn, int(num_heads), int(head_dim), int(num_kv), causal=True
        )
        layer.self_attn = te_mod
        swapped += 1

    # 注入根引用（供前向读取 _te_packing_ctx）
    for m in model.modules():
        if isinstance(m, TEPackedSelfAttention):
            m._te_root = model
    return swapped



def _get_parent(root: nn.Module, child: nn.Module):
    for name, module in root.named_modules():
        for cname, cm in module.named_children():
            if cm is child:
                return module
    return None

def _replace_child(parent: nn.Module, old: nn.Module, new: nn.Module):
    for cname, cm in list(parent.named_children()):
        if cm is old:
            setattr(parent, cname, new)
            return
    raise RuntimeError("replace child failed")

# -------------------------
# 3) 训练脚本主体（Accelerate FP8 + DeepSpeed ZeRO3）
# -------------------------
def main():
    # ★ TE FP8 配方：开启 FP8 注意力（cuDNN 子后端2）
    te_kwargs = TERecipeKwargs(
        fp8_format="HYBRID",            # FWD:E4M3, BWD:E5M2（常见组合）
        amax_history_len=32,
        amax_compute_algo="max",
        use_autocast_during_eval=False,
    )

    ds_plugin = DeepSpeedPlugin(
        zero_stage=ZERO_STAGE,
        zero3_init_flag=(ZERO_STAGE == 3),
    )
    accelerator = Accelerator(
        mixed_precision="fp8",
        kwargs_handlers=[te_kwargs],
        gradient_accumulation_steps=GRAD_ACCUM,
        deepspeed_plugin=ds_plugin,
    )

    # ---- B) TE 自己的 recipe：把 fp8_dpa 开关写在这 ----
    fp8_recipe = DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=32,
        amax_compute_algo="max",
    )
    # 关键：启用 cuDNN 子后端2的 FP8 注意力（DPA）
    fp8_recipe.fp8_dpa = True  # ★ 开启 FP8 attention
    # 可选：实验特性，去掉 DPA 前后 cast
    # fp8_recipe.fp8_mha = True

    accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = BATCH_SIZE
    if ZERO_STAGE == 3:
        accelerator.state.deepspeed_plugin.deepspeed_config.setdefault(
            "zero_optimization", {}
        ).setdefault("stage3_gather_16bit_weights_on_model_save", True)

    torch.backends.cuda.matmul.allow_tf32 = True

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # dataset
    ds = load_from_disk(str(TOK_PATH))
    collator = TEPackingCollator(pad_id=tokenizer.pad_token_id, pack_to=PACK_TO)
    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, num_workers=8, pin_memory=True, persistent_workers=True
    )

    # model（权重 bf16；注意力由 TE 处理 FP8，线性层 FP8 由 TE/Accelerate recipe 统一管理）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # ★ 把注意力替换成 TE 版本
    swapped = _swap_attn_with_te(model)
    accelerator.print(f"[TE Attention] swapped modules: {swapped}")

    from types import MethodType
    wrapped = contextual_fp8_autocast(model.forward, fp8_recipe=fp8_recipe, use_during_eval=False)
    model.forward = MethodType(wrapped, model)  # ★ 绑定 self，避免 missing 'self'

    # optimizer / scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.0)

    # ★ 交给 Accelerate
    model, optimizer, dl = accelerator.prepare(model, optimizer, dl)

    num_update_steps_per_epoch = math.ceil(len(dl) / 1)
    total_steps  = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        for batch in dl:
            # ★ 把当步的 varlen 上下文挂到根模型，供注意力读取
            model._te_packing_ctx = {
                "cu_seqlens": batch["cu_seqlens"].to(accelerator.device, non_blocking=True),
                "max_seqlen": int(batch["max_seqlen"]),
            }
            with accelerator.accumulate(model):
                # ★ 常规前向：Attention 内部会从 _te_packing_ctx 取 cu_seqlens
                outputs = model(
                    input_ids=batch["input_ids"].to(accelerator.device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(accelerator.device, non_blocking=True),
                    labels=batch["labels"].to(accelerator.device, non_blocking=True),
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % LOG_EVERY == 0:
                accelerator.print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(OUTPUT_DIR, safe_serialization=True)
        tokenizer.save_pretrained(OUTPUT_DIR)
        accelerator.print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
