#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_ID = "unsloth/gpt-oss-20b-BF16"  # 如果实际 repo 名不是这个，把它改成你要的 HF repo id

def main():
    # 0) 基本环境信息
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
        print("total vram (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    # 1) 先只加载 config：几乎不占显存/内存，也能看“结构超参”
    print("\n==== Load config only ====")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(cfg)

    # 2) 决定 dtype（尽量省显存）
    # - 有 A100/H100 等支持 bf16：bf16 更稳
    # - 否则用 fp16
    if torch.cuda.is_available():
        # bf16 需要硬件支持；这里用一个保守判断
        major, _ = torch.cuda.get_device_capability(0)
        use_bf16 = major >= 8  # Ampere(8.0/8.6)及以上通常OK
        dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        dtype = torch.float32

    print("\nSelected dtype:", dtype)

    # 3) 加载 tokenizer（有些模型需要 trust_remote_code）
    print("\n==== Load tokenizer ====")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("tokenizer:", tok.__class__.__name__)
    print("vocab_size:", getattr(tok, "vocab_size", None))

    # 4) 加载模型（会很大）
    # device_map="auto" 需要 accelerate；一般 transformers 会自动提示安装
    # low_cpu_mem_usage=True 省内存
    print("\n==== Load model ====")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    # 5) 打印模型结构（最直观）
    print("\n==== print(model) ====")
    print(model)

    # 6) 打印关键结构信息
    print("\n==== Model class / config summary ====")
    print("model class:", model.__class__.__name__)
    print("architectures:", getattr(model.config, "architectures", None))
    for k in ["hidden_size", "n_embd", "num_hidden_layers", "n_layer", "num_attention_heads", "n_head",
              "intermediate_size", "vocab_size", "max_position_embeddings", "rope_theta"]:
        if hasattr(model.config, k):
            print(f"{k}: {getattr(model.config, k)}")

    # 7) 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n==== Params ====")
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")

    # 8) 打印前 N 个参数名 + shape（看模块组织很有用）
    print("\n==== Named parameters (first 40) ====")
    for i, (name, p) in enumerate(model.named_parameters()):
        print(f"{i:04d}  {name:80s}  {tuple(p.shape)}  dtype={p.dtype}  device={p.device}")
        if i >= 39:
            break

    # 9) 打印前 N 个子模块（看层级结构）
    print("\n==== Named modules (first 60) ====")
    for i, (name, m) in enumerate(model.named_modules()):
        print(f"{i:04d}  {name:80s}  {m.__class__.__name__}")
        if i >= 59:
            break

if __name__ == "__main__":
    # 可选：减少 HF 并行下载时的一些 warning
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
