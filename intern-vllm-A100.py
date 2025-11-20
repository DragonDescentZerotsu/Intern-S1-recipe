from transformers import AutoProcessor, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

model_name = "internlm/Intern-S1"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

llm = LLM(
    model=model_name,
    dtype="bfloat16",                 # 或 "auto"
    tensor_parallel_size=8,           # ← 这里打开 TP
    trust_remote_code=True,
    gpu_memory_utilization=0.92,      # 合理提升吞吐
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "tell me about an interesting physical phenomenon."},
        ],
    }
]

prompts = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(dtype=torch.bfloat16)
# print(prompts)
# exit(0)
sp = SamplingParams(max_tokens=1024, temperature=0.7, top_p=0.95)
outputs = llm.generate(prompts, sp)

for i, out in enumerate(outputs):
    # 每个 out 可能有多条候选，这里取第一条
    print(f"[Sample {i}] {out.outputs[0].text.strip()}")