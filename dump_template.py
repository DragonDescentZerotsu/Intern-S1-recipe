
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)
with open("template.jinja", "w") as f:
    f.write(tokenizer.chat_template)
