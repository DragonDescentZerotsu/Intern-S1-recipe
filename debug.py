from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import torch

model_name = "internlm/Intern-S1-mini"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True
)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

messages = [
    {"role": "user", "content": "<IMG_CONTEXT>\nPlease describe the image explicitly."}
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

out = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
