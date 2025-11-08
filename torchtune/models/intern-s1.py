# models/interns1_model.py
from __future__ import annotations
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor


class InternS1ForSFT(nn.Module):
    """
    轻量封装：内部持有 HF 模型与 processor。
    前向接受 dataset 的 collate 输出（input_ids/attention_mask/...），回传 loss。
    """
    def __init__(
        self,
        model_name: str = "internlm/Intern-S1-mini",
        torch_dtype: str = "bfloat16",
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
                target = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target, lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
                self.model = get_peft_model(self.model, cfg)
            except Exception as e:
                raise RuntimeError(f"PEFT/LoRA 未安装或初始化失败: {e}")

    @property
    def hf_processor(self):
        return self.processor

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(**batch, use_cache=False)
        # HF causal LM 在传入 labels 时会返回 loss
        return {"loss": outputs.loss}

def interns1_model(**kwargs):
    """
    builder 函数（YAML 用 _component_ 引用）
    只暴露少量关键开关，隐藏底层构造细节（官方推荐的 builder 模式）。
    """
    return InternS1ForSFT(**kwargs)
