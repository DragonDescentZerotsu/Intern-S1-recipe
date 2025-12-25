
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)
print(tokenizer.chat_template)
