# 选择显卡
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

from pathlib import Path

# 1) 读取 Intern-S1 全模型
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoConfig, AutoTokenizer
import torch

current_dir = Path(__file__).parent.resolve()

ckpt = "internlm/Intern-S1-mini"  # 你也可以固定到特定 commit id 以锁定版本
# ckpt = "internlm/Intern-S1"
# ckpt = "internlm/Intern-S1-mini-FP8"
full = AutoModelForCausalLM.from_pretrained(
    ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16
)

# 2) 拿到文本子网和其 text_config（包含正确 vocab_size 与 MoE 配置）
lm_base = full.language_model                      # 这是“无 lm_head 的”decoder基座
text_cfg = full.config.text_config                 # 就是 Qwen3-MoE 的配置（含 A22B MoE）
                                                   # ↑ 这些字段名由官方实现提供
print('full model:')
print(full)
print('lm_base:')
print(lm_base)
# exit(1)
# 3) 构建一个“标准的”CausalLM，并把基座权重 + lm_head 一起灌进去
clm = AutoModelForCausalLM.from_config(text_cfg)   # 得到 Qwen3-MoE ForCausalLM
print('clm:')
print(clm)
# exit(1)
missing, unexpected = clm.model.load_state_dict(lm_base.state_dict(), strict=False)
# 把头也拷过来（Intern-S1 的 lm_head 在外层）
with torch.no_grad():
    clm.lm_head.weight.copy_(full.lm_head.weight)  # tied或untied都覆盖到

# 4) 保存纯文本模型
clm.save_pretrained(current_dir.parent/"checkpoints"/"megatron"/"hf_version"/"intern_s1_mini_text_llm")
# clm.save_pretrained(current_dir.parent/"checkpoints"/"megatron"/"hf_version"/"intern_s1_text_llm")
# clm.save_pretrained(current_dir.parent/"checkpoints"/"megatron"/"hf_version"/"intern_s1_mini_FP8_text_llm")

# 5) 保存 Intern-S1 自己的 tokenizer（含新增 tokens）
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=True)
# tok.save_pretrained("./interns1_tokenizer")

# 6) 小检查：词表大小要一致
assert tok.vocab_size == clm.get_input_embeddings().weight.size(0)