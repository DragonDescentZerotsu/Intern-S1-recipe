import os
import sys

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import multiprocessing as mp

# 1) Force vLLM to use spawn
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_USE_V1", "1")
mp.set_start_method("spawn", force=True)

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, classification_report
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from utils.TDC_answer_parser import extract_answer, parse_answer

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'  # TODO: device GPU #


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Test TDC tasks with vLLM using preprocessed CoT data (F1 Score)')

    parser.add_argument('--model-path', type=str,
                        default='openai/gpt-oss-120b',
                        # zai-org/GLM-4.7-Flash
                        # internlm/Intern-S1-mini
                        # openai/gpt-oss-120b
                        # openai/gpt-oss-20b
                        help='Path to the model checkpoint')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA adapter')
    parser.add_argument('--lora-path', type=str,
                        default="",
                        help='Path to LoRA adapter')
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parent.parent / "DataPrepare/TDC_test_prompts_label_scaffold",
                        help='Directory containing preprocessed test data')
    parser.add_argument('--task-groups', nargs='+',
                        default=['ADME', 'Tox', 'HTS'],
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding',
                                 'TrialOutcome', 'PeptideMHC', 'all'],
                        help='Task groups to run')
    parser.add_argument('--max-model-len', type=int, default=1024 * 30, help='Max model length')
    parser.add_argument('--tensor-parallel-size', type=int, default=2, help='Tensor parallel size')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.92, help='GPU memory utilization')
    parser.add_argument('--max-num-seqs', type=int, default=256, help='Max number of sequences')
    
    # Sampling arguments
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')  # 0.0 ?
    parser.add_argument('--max-tokens', type=int, default=1024 * 29, help='Max new tokens to generate')

    parser.add_argument('--strip-smiles-tags', action='store_false', help='Remove <SMILES> and </SMILES> from prompts before inference')
    parser.add_argument('--log-file', action='store_false', help='Enable logging to file')
    parser.add_argument('--log-file-name', type=str, default="no_Tools_gpt-oss-120b_{model_name}_{t_stamp}_4.log", help='Log file name pattern')  # TODO: log file name 

    args = parser.parse_args()
    return args


def load_tasks_map(data_dir):
    """
    Maps task groups to specific files in the data directory.
    """
    mapping = {
        'Tox': [
            # 'hERG_Karim.jsonl',  # 2690
            # 'Carcinogens_Lagunin.jsonl',  # 56
            # 'Skin_Reaction.jsonl',  # 82
            # 'hERG.jsonl',  # 132
            # 'DILI.jsonl',  # 96
            # 'ClinTox.jsonl',  # 297
            # 'AMES.jsonl',  # 1457
            # 'Tox21.jsonl',  # 15584
            # 'herg_central_hERG_inhib.jsonl',  # 61379
            # -----------------------------------------
            # 'ToxCast.jsonl'  # 307282
        ],
        'ADME': [
            # 'PAMPA_NCATS.jsonl',  # 408
            # 'HIA_Hou.jsonl',  # 117
            # 'BBB_Martins.jsonl',  # 406
            # 'Pgp_Broccatelli.jsonl',  # 245
            # 'Bioavailability_Ma.jsonl',  # 128
            # 'CYP2C9_Substrate_CarbonMangels.jsonl',  # 135
            # 'CYP2D6_Substrate_CarbonMangels.jsonl',  # 135
            # 'CYP3A4_Substrate_CarbonMangels.jsonl',  # 135
            # 'CYP1A2_Veith.jsonl',  # 2517
            # 'CYP2C19_Veith.jsonl',  # 2534
            # 'CYP2C9_Veith.jsonl',  # 2419
            # 'CYP2D6_Veith.jsonl',  # 2626
            # 'CYP3A4_Veith.jsonl',  # 2467
        ],
        'HTS': [
            'HIV.jsonl',  # 8225
            'SARSCoV2_3CLPro_Diamond.jsonl',  # 176
            'SARSCoV2_Vitro_Touret.jsonl',  # 298
            # -----------------------------------------
            # 'butkiewicz.jsonl'  # 401997
        ],
        'Develop': ['SAbDab_Chen.jsonl'],  # 482
        'PPI': ['HuRI.jsonl'],  # 20282
        'TrialOutcome': [
            'phase1.jsonl',
            'phase2.jsonl',
            'phase3.jsonl'
        ],
        'PeptideMHC': [
            'MHC1_IEDB-IMGT_Nielsen.jsonl',  # 37197
            'MHC2_IEDB_Jensen.jsonl'  # 26856
        ]
    }

    # Check what actually exists in the directory
    available_files = set(f.name for f in Path(data_dir).glob("*.jsonl"))

    final_map = {}
    for group, filenames in mapping.items():
        existing = [f for f in filenames if f in available_files]
        if existing:
            final_map[group] = existing

    return final_map


def to_prompt_user_block(text: str, tokenizer, model_name: str = None) -> str:
    """
    Apply chat template.
    """
    messages = []
    kwargs = dict(tokenize=False, add_generation_prompt=True)

    if model_name and 'Nemotron' in model_name:
        messages.append({"role": "system", "content": "/think"})  # 打开 think 模式
        messages.append({"role": "user", "content": text})
    elif model_name and ("Qwen" in model_name or "Intern" in model_name):
        kwargs["enable_thinking"] = True   # 打开 think 模式
        messages.append({"role": "user", "content": text})
    else:
        messages.append({"role": "user", "content": text})
    
    return tokenizer.apply_chat_template(messages, **kwargs)


def strip_smiles_tags(text: str) -> str:
    return text.replace("<SMILES>", "").replace("</SMILES>", "")


def main():
    args = get_args()

    # Configure File Handler if log_file is provided
    if args.log_file:
        current_dir = Path(__file__).parent.resolve()
        log_dir = current_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_safe = Path(args.model_path).name
        log_path = log_dir / args.log_file_name.format(model_name=model_name_safe, t_stamp=timestamp)
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    # ===================== Model & Tokenizer =====================
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        llm = LLM(
            model=args.model_path,
            # enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt={"video": 0, "image": 0},
            enable_lora=args.use_lora,
        )
    except RuntimeError as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        return

    lora_req = None
    if args.use_lora and args.lora_path and os.path.isdir(args.lora_path):
        lora_req = LoRARequest(
            lora_name='lora',
            lora_int_id=1,
            lora_path=args.lora_path
        )

    # ===================== Sampling Params =====================
    # Greedy decoding for F1 (temp=0)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        detokenize=True,
    )

    # ===================== Data Loading =====================
    data_path = args.data_dir
    if not data_path.exists():
        logger.error(f"Data directory {data_path} does not exist.")
        return

    task_map = load_tasks_map(data_path)

    groups_to_run = args.task_groups
    if 'all' in groups_to_run:
        groups_to_run = list(task_map.keys())

    logger.info(f"Running groups: {groups_to_run}")

    all_results = {}

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group}")
            continue

        files = task_map[group]
        all_results[group] = {}

        for filename in files:
            file_path = data_path / filename
            task_name = filename.replace('.jsonl', '')

            logger.info(f"Processing Task: {task_name}")

            # Load test data
            test_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))

            if not test_data:
                logger.warning(f"No data in {filename}")
                continue

            logger.info(f"Total test samples: {len(test_data)}")

            # ===================== Prepare Prompts =====================
            base_prompts = []
            test_labels = []
            
            for item in test_data:
                raw_text = item['text'] # Already contains CoT instructions
                
                if args.strip_smiles_tags:
                    raw_text = strip_smiles_tags(raw_text)

                prompt = to_prompt_user_block(raw_text, tokenizer, model_name=args.model_path)
                base_prompts.append(prompt)
                test_labels.append(int(item['Y']))

            # ===================== Inference =====================
            logger.info(f"Running inference on {len(base_prompts)} samples...")
            
            outputs = llm.generate(base_prompts, sp, lora_request=lora_req)
            
            preds = []
            valid_labels = []
            failed_parses_count = 0
            
            for i, out in enumerate(outputs):
                txt = out.outputs[0].text
                
                # Use utils function to extract answer
                ans_txt, fmt_ok = extract_answer(txt)
                # Parse answer (A->0, B->1)
                # Note: We enable thinking support in parse_answer if needed, assuming instructions are followed
                pred = parse_answer(ans_txt, fmt_ok, think_is_on=True)
                
                if pred not in (0, 1):
                    failed_parses_count += 1
                    pred = 1 - test_labels[i]   # 强制给反标签，必错且保持二分类

                preds.append(pred)
                
                valid_labels.append(test_labels[i])
                
                if i < 3: # Debug print first few
                    logger.info(f"Sample {i} | Label: {test_labels[i]} | Pred: {pred}")
                    logger.info(f"\nOutput snippet: \n{txt}\n")

            # ===================== Evaluation =====================
            logger.info("\n" + "=" * 80)
            logger.info(f"{task_name} EVALUATION RESULTS (F1 Score)")
            logger.info("=" * 80)
            logger.info(f"Failed parses: {failed_parses_count}/{len(base_prompts)}")
            
            # Remove invalid predictions (-1) only if we want to ignore them, 
            # BUT typically for a fair comparison we should count them as errors.
            # However, scikit-learn will complain if labels are not in [0, 1].
            # Let's map -1 to the opposite of truth or keep it to penalize.
            # Actually, f1_score with labels=[0, 1] and pos_label=1 will treat others as mismatch.
            
            score = f1_score(valid_labels, preds, average='macro', pos_label=1)
            logger.info(f"Classification Report:\n{classification_report(valid_labels, preds, digits=4, labels=[0, 1])}")
            logger.info(f"F1 Score: {score:.4f}")
            
            all_results[group][task_name] = score
            logger.info("=" * 80 + "\n")

    # ===================== Summary =====================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info("")

    for group in groups_to_run:
        if group not in all_results or not all_results[group]:
            continue
        logger.info(f"\n{group} Tasks:")
        logger.info("-" * 40)

        group_scores = []
        for task_name, score in all_results[group].items():
            logger.info(f"  {task_name}: {score:.4f}")
            group_scores.append(score)

        if group_scores:
            logger.info(f"  Average: {np.mean(group_scores):.4f}")

    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
