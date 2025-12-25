from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import requests
import torch
from vllm.lora.request import LoRARequest
from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import json
from utils import normalize_DeepSeek_V32_messages

# Configuration
MODEL_NAME = "internlm/Intern-S1-mini-FP8"
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = 50
MAX_TOKENS = 1024 * 10

def main():

    tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)  # for Debug use
    # rendered = tokenizer.apply_chat_template(normalized_messages, tokenize=False, enable_thinking=True, add_generation_prompt=False)

    pickle_path = "agent/DeepSeek_V32_output/skin_reaction_1.pkl"
    with open(pickle_path, "rb") as f:
        messages = pickle.load(f)
    # print(messages)
    
    normalized_messages = normalize_DeepSeek_V32_messages(messages)
    print(json.dumps(normalized_messages, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
