import argparse
import os
import math
import torch
from vllm import LLM, SamplingParams

def calculate_ppl_vllm(MODEL_NAME, devices, sentences):
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        print(f"Setting CUDA_VISIBLE_DEVICES={devices}")

    # Initialize vLLM
    # Note: score method might not need SamplingParams in the same way, 
    # but we initialize LLM as usual.
    print(f"Loading model from {MODEL_NAME}...")

    TEMPERATURE = 0.7
    TOP_P = 1.0
    TOP_K = 50
    USE_LORA = False

    if 'Intern-S1-mini' in MODEL_NAME:
        TENSOR_PARALLELE_SIZE = 1
    elif 'Intern-S1-FP8' in MODEL_NAME:
        TENSOR_PARALLELE_SIZE = 4  # 适配 H100
    elif 'Intern-S1' in MODEL_NAME:
        TENSOR_PARALLELE_SIZE = 8  # 适配 A100

    llm = LLM(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=1024 * 24,
        max_num_batched_tokens=1024 * 24,
        quantization="fp8" if 'FP8' in MODEL_NAME else None,  # "fp8",          # 触发 FP8 W8A8 路径（或自动识别 FP8 检查点）
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLELE_SIZE,  # 1 if 'Intern-S1-mini' in MODEL_NAME else 8,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_num_seqs=256,
        limit_mm_per_prompt={"video": 0, "image": 0},
        enable_lora=USE_LORA,
        tokenizer_mode="auto"
    )

    print("-" * 50)
    
    # Use the generate method with prompt_logprobs
    # We set max_tokens=1 because we only care about the prompt logprobs.
    # vLLM 0.11.0 has a limit of 20 for prompt_logprobs.
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=20)
    
    outputs = llm.generate(sentences, sampling_params)

    for i, output in enumerate(outputs):
        sentence = sentences[i]
        print(f"Sentence {i+1}: {sentence}")
        
        prompt_logprobs = output.prompt_logprobs
        token_ids = output.prompt_token_ids
        
        total_logprob = 0.0
        count = 0
        missing_tokens = 0
        
        # prompt_logprobs[i] is the logprob of token_ids[i] given token_ids[:i]
        # prompt_logprobs[0] is None.
        
        for j in range(1, len(token_ids)):
            token_id = token_ids[j]
            logprobs_dict = prompt_logprobs[j]
            
            if token_id in logprobs_dict:
                total_logprob += logprobs_dict[token_id].logprob
                count += 1
            else:
                # Token not in top K. 
                # Use min logprob as estimate.
                min_logprob = min(lp.logprob for lp in logprobs_dict.values())
                total_logprob += min_logprob
                missing_tokens += 1
        
        if count > 0:
            avg_logprob = total_logprob / (count + missing_tokens)
            ppl = math.exp(-avg_logprob)
            print(f"PPL {i+1}: {ppl}")
            if missing_tokens > 0:
                print(f"Warning: {missing_tokens} tokens were not in top 20 logprobs. PPL is an estimate.")
        else:
            print("Could not calculate PPL.")
        
        print("-" * 50)

    return outputs

def main():
    parser = argparse.ArgumentParser(description="Calculate PPL for sentences using vLLM.")
    parser.add_argument("--model_path", type=str, default="internlm/Intern-S1-FP8", 
                        help="Path to the model or HuggingFace model ID.")
    parser.add_argument("--devices", type=str, default=None, 
                        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    args = parser.parse_args()

    sentences = [
        "<SMILES>CC1=CC(=NO1)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES> is the SMILES of sulfamethoxazole.",
        "<SMILES>Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1</SMILES> is the SMILES of sulfamethoxazole.",
        "<SMILES>CC1=NOC(NS(=O)(=O)C2=CC=C(N)C=C2)=C1</SMILES> is the SMILES of sulfamethoxazole.",
        "<SMILES>CC1=NOC(NS(=O)(=O)C2=CC=C(N)C=C2)=C1</SMILES> is the SMILES of sulfisoxazole.",
        "<SMILES>CC1=C(ON=C1C)NS(=O)(=O)C2=CC=C(C=C2)N</SMILES> is the SMILES of sulfisoxazole."
    ]

    outputs = calculate_ppl_vllm(args.model_path, args.devices, sentences)
    
    for i, output in enumerate(outputs):
        # The structure of ScoringRequestOutput usually contains 'logprobs'
        # Let's assume it has a way to get the total logprob or we sum it.
        # Inspecting the object:
        # It likely has 'logprobs' which is a list of logprobs for the sequence.
        
        # Let's try to access logprobs.
        # If it fails, we will see the error and fix it.
        
        # NOTE: In some versions, output might be just the score (float).
        # But usually it's an object.
        
        # Let's try to be robust.
        if hasattr(output, 'logprobs'):
            # output.logprobs is likely a list of something.
            # We want the logprob of the token that was actually in the sequence.
            # For 'score' mode, it should return exactly that.
            
            # Let's sum them up.
            # Note: The first token might be None or BOS.
            
            # Actually, let's just print the object attributes for the first run to be sure.
            print(f"Output {i+1} type: {type(output)}")
            print(f"Output {i+1} dir: {dir(output)}")
            
            # But to be useful immediately, let's try to calculate.
            # If output is ScoringRequestOutput, it might have 'outputs' which is a list of ScoringOutput?
            # No, 'score' returns a list of ScoringRequestOutput.
            pass

if __name__ == "__main__":
    main()
