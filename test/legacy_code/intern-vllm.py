from transformers import AutoProcessor, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

model_name = "internlm/Intern-S1-FP8"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

llm = LLM(
    model=model_name,
    quantization="fp8",          # 触发 FP8 W8A8 路径（或自动识别 FP8 检查点）
    dtype="bfloat16",            # 非 FP8 运算与累加精度（与 FP8 内核配套）
    tensor_parallel_size=4,      # Tensor Parallel（按你的卡数调整）
    kv_cache_dtype="fp8_e5m2",   # H100 上常用的 KV Cache 精度
    trust_remote_code=True,
    gpu_memory_utilization=0.92,
    # 如果你的版本支持，可开启 KV 动态缩放（否则需从检查点提供缩放因子）
    # calculate_kv_scales=True,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "tell me about an interesting physical phenomenon."},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=32768)
decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
print(decoded_output)
