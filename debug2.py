from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from PIL import Image
import requests
import torch
from vllm.lora.request import LoRARequest
from rdkit import Chem
from rdkit.Chem import Draw

# Configuration
MODEL_NAME = "internlm/Intern-S1-mini-FP8"
TEMPERATURE = 0.7
TOP_P = 1.0
TOP_K = 50
MAX_TOKENS = 1024 * 10

def main():
    # 1. Initialize LLM
    print("Initializing vLLM...")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=1024 * 24,
        max_num_batched_tokens=1024 * 24,
        enforce_eager=True,
        quantization="fp8" if 'FP8' in MODEL_NAME else None,
        trust_remote_code=True,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.92,
        max_num_seqs=256,
        dtype="bfloat16"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 2. Prepare SamplingParams
    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS
    )

    # 3. Load Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(f"Loading image from {url}...")
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    entry = 'CC1=C(ON=C1C)NS(=O)(=O)C2=CC=C(C=C2)N'
    mol = Chem.MolFromSmiles(entry)
    if mol:
        image = Draw.MolToImage(mol, size=(224, 224))
    # Resize image if needed, similar to Draw.MolToImage(mol, size=(224, 224)) in reference
    # but for natural images usually we keep valid resolution or resize to model supported res.
    # InternVL usually handles resizing internally or via processor, but vLLM expects PIL image.
    # We will pass the PIL image directly.

    # 4. Construct Prompt
    # <IMG_CONTEXT> token is used in the reference script prompt construction
    prompt_text = "<IMG_CONTEXT>\nWhat's the SMIELS string of the molecules in this image?"
    
    messages = [
        {"role": "user", "content": prompt_text}
    ]
    
    # Use tokenizer to apply chat template to get the raw prompt string
    final_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    print("final_prompt: ", final_prompt)

    # 5. Prepare Inputs
    # vLLM generate accepts list of dicts with 'prompt' and 'multi_modal_data' keys
    inputs = [
        {
            "prompt": final_prompt,
            "multi_modal_data": {"image": image}
        }
    ]

    # 6. Generate
    print("Generating response...")
    outputs = llm.generate(inputs, sp)

    # 7. Print Output
    for output in outputs:
        generated_text = output.outputs[0].text
        print("\nGenerated Description:")
        print(generated_text)

if __name__ == "__main__":
    main()
