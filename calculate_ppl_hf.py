import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import math
import sys
import argparse
import os

print(f"Python executable: {sys.executable}")


def calculate_ppl(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    ppl = torch.exp(loss)
    return ppl.item()

def main():
    parser = argparse.ArgumentParser(description="Calculate PPL for sentences using a causal LM.")
    parser.add_argument("--model_path", type=str, default="internlm/Intern-S1-mini", 
                        help="Path to the model or HuggingFace model ID.")
    parser.add_argument("--devices", type=str, default=None, 
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    args = parser.parse_args()

    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        print(f"Setting CUDA_VISIBLE_DEVICES={args.devices}")

    model_path = args.model_path
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto"
    )

    sentences = [
        "<SMILES>CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES>", # is the SMILES of sulfamethoxazole.",
        "<SMILES>Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1</SMILES>", # is the SMILES of sulfamethoxazole.",
        "<SMILES>CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES>is the SMILES of sulfamethoxazole.",
        "<SMILES>Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1</SMILES> is the SMILES of sulfamethoxazole."
    ]

    # sentences = [
    #     "<SMILES>CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES> is the SMILES of sulfamethoxazole.",
    #     "<SMILES>Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1</SMILES> is the SMILES of sulfamethoxazole.",
    #     "<SMILES>CC1=NOC(NS(=O)(=O)C2=CC=C(N)C=C2)=C1</SMILES> is the SMILES of sulfamethoxazole.",
    #     "<SMILES>CC1=NOC(NS(=O)(=O)C2=CC=C(N)C=C2)=C1</SMILES> is the SMILES of sulfisoxazole.",
    #     "<SMILES>CC1=C(ON=C1C)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES> is the SMILES of sulfisoxazole."
    # ]

    for i, sentence in enumerate(sentences):
        print("-" * 50)
        print(f"Sentence {i+1}: {sentence}")
        ppl = calculate_ppl(model, tokenizer, sentence)
        print(f"PPL {i+1}: {ppl}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()
