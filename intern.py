from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model_name = "internlm/Intern-S1"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True)

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
