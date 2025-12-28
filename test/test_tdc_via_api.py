import os
import argparse
import json
import logging
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import numpy as np
from sklearn.metrics import roc_auc_score
from openai import OpenAI

# Add project root to path to import utils/tools if needed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Verify these imports exist in your project structure
try:
    from tools import * 
    from utils.TDC_answer_parser import extract_answer, parse_answer
except ImportError:
    # Fallback if specific tools/utils are not easily importable or if running standalone
    # We will define a simple extractor if import fails, but prefer project utils
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)  # 隐藏 HTTP 请求日志
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Run TDC benchmark tasks via OpenAI-compatible API')
    
    parser.add_argument('--task-groups', nargs='+', default=['ADME'],
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'Other', 'all'],
                        help='Task groups to run')
    parser.add_argument('--n-samples', type=int, default=16, help='Number of samples per query')
    parser.add_argument('--api-base', type=str, default="http://0.0.0.0:8000/v1", help='API Base URL')
    parser.add_argument('--api-key', type=str, default="EMPTY", help='API Key')
    parser.add_argument('--model', type=str, default="", help='Model name (optional, will query server if empty)')
    parser.add_argument('--num-processes', type=int, default=32, help='Number of parallel workers')
    parser.add_argument('--data-dir', type=str, default="DataPrepare/TDC_test_prompts_label", help='Directory containing processed test data')
    parser.add_argument('--thinking', action='store_false', help='Enable thinking parameter for DeepSeek models')  # TODO: 注意这里 thinking 到底是开了还是没开
    
    args = parser.parse_args()
    return args

def run_turn(client, messages, model_name, thinking=False):
    """
    Executes a single turn of conversation. 
    Simplified version of agent/skin-reaction.py's run_turn, 
    assuming single-turn or limited tool usage if tools are enabled.
    For standard TDC evaluation, it's often single-turn QA.
    """
    try:
        extra_body = {}
        if thinking:
             extra_body = {"chat_template_kwargs": {"thinking": True}} # vLLM Server style

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # tools=TOOLS if 'TOOLS' in globals() else None, # Enable tools if needed
            extra_body=extra_body
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in chat completion: {e}")
        return None

def worker_process_sample(args):
    """
    Args: (index, text, label_dummy, api_base, api_key, model_name, thinking)
    Returns: (index, prediction_int_or_None)
    """
    index, text, _, api_base, api_key, model_name, thinking = args
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    
    messages = [{'role': 'user', 'content': text}]
    
    try:
        response_text = run_turn(client, messages, model_name, thinking)
        if response_text:
            # parsing logic
            # Ensure extract_answer and parse_answer are available
            from utils.TDC_answer_parser import extract_answer, parse_answer
            
            ans_txt, fmt_ok = extract_answer(response_text)
            prediction = parse_answer(ans_txt, fmt_ok, think_is_on=thinking) # Assuming thinking affects parsing logic?
            # Note: parse_answer signature might vary. 
            # In skin-reaction.py: parse_answer(ans_txt, fmt_ok, think_is_on=True)
            
            return (index, prediction)
    except Exception as e:
        # logger.error(f"Worker error: {e}")
        pass
        
    return (index, None)

def load_tasks_map(data_dir):
    """
    Maps task groups to specific files in the data directory.
    This mapping relies on the file naming convention from process_tdc_test.py.
    """
    mapping = {
        'Tox': ['Tox21.jsonl', 'ToxCast.jsonl', 'Skin_Reaction.jsonl', 'hERG.jsonl', 'AMES.jsonl', 'DILI.jsonl', 'ClinTox.jsonl', 'herg_central.jsonl'],
        'ADME': ['PAMPA_NCATS.jsonl', 'HIA_Hou.jsonl', 'Bioavailability_Ma.jsonl', 'BBB_Martins.jsonl', 'Pgp_Broccatelli.jsonl', 
                 'CYP1A2_Veith.jsonl', 'CYP2C19_Veith.jsonl', 'CYP2C9_Veith.jsonl', 'CYP2D6_Veith.jsonl', 'CYP3A4_Veith.jsonl',
                 'CYP2C9_Substrate_CarbonMangels.jsonl', 'CYP2D6_Substrate_CarbonMangels.jsonl', 'CYP3A4_Substrate_CarbonMangels.jsonl'],
        'HTS': ['HIV.jsonl', 'SARSCoV2_3CLPro_Diamond.jsonl', 'SARSCoV2_Vitro_Touret.jsonl', 'butkiewicz.jsonl'],
        'Develop': ['SAbDab_Chen.jsonl'],
        'PPI': ['HuRI.jsonl'],
        'TCREpitopeBinding': ['Weber.jsonl'],
        'TrialOutcome': ['phase1.jsonl', 'phase2.jsonl', 'phase3.jsonl'],
        'PeptideMHC': ['MHC1_IEDB_IMGT_Nielsen.jsonl', 'MHC2_IEDB_Jensen.jsonl']
    }
    
    # Check what actually exists in the directory
    available_files = set(f.name for f in Path(data_dir).glob("*.jsonl"))
    
    final_map = {}
    for group, filenames in mapping.items():
        existing = [f for f in filenames if f in available_files]
        if existing:
            final_map[group] = existing
            
    return final_map

def main():
    args = get_args()
    
    data_path = Path(args.data_dir).resolve()
    if not data_path.exists():
        logger.error(f"Data directory {data_path} does not exist.")
        return

    # Determine Model Name if not provided
    if not args.model:
        try:
            temp_client = OpenAI(api_key=args.api_key, base_url=args.api_base)
            models = temp_client.models.list()
            args.model = models.data[0].id
            logger.info(f"Detected model: {args.model}")
        except Exception as e:
            logger.error(f"Could not connect to API to detect model: {e}")
            return

    task_map = load_tasks_map(data_path)
    
    groups_to_run = args.task_groups
    if 'all' in groups_to_run:
        groups_to_run = list(task_map.keys())

    logger.info(f"Running groups: {groups_to_run}")

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group} or group not defined.")
            continue
            
        files = task_map[group]
        
        for filename in files:
            file_path = data_path / filename
            task_name = filename.replace('.jsonl', '')
            
            logger.info(f"Processing Task: {task_name}")
            
            # Load Data
            raw_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
            
            if not raw_data:
                logger.warning(f"No data in {filename}")
                continue

            # Prepare args for multiprocessing
            # tasks[i] = (index, text, label, ...)
            worker_tasks = []
            for idx, item in enumerate(raw_data):
                task_tuple = (idx, item['text'], item['Y'], args.api_base, args.api_key, args.model, args.thinking)
                for _ in range(args.n_samples):
                    worker_tasks.append(task_tuple)
            
            # Execute
            results = []
            with multiprocessing.Pool(processes=args.num_processes) as pool:
                for result in tqdm(pool.imap_unordered(worker_process_sample, worker_tasks), total=len(worker_tasks), desc=f"{task_name}"):
                    results.append(result)
            
            # Aggregate
            item_results = {} # idx -> list of predictions
            for idx, pred in results:
                if idx not in item_results:
                    item_results[idx] = []
                if pred is not None:
                    item_results[idx].append(pred)
            
            # Calculate metrics
            y_true = []
            y_scores = []
            
            for idx in range(len(raw_data)):
                if idx not in item_results or not item_results[idx]:
                    continue
                
                preds = item_results[idx]
                count_0 = preds.count(0)
                count_1 = preds.count(1)
                total_valid = count_0 + count_1
                
                if total_valid == 0:
                    continue
                
                score = count_1 / total_valid
                y_true.append(int(raw_data[idx]['Y'])) # ensure int
                y_scores.append(score)
            
            # Final Report for this Task
            if len(set(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_scores)
                logger.info(f"{task_name} AUROC: {auroc:.4f}")
            else:
                logger.info(f"{task_name} AUROC: Cannot calculate (one class or empty)")
                # Print mean score if AUROC fails
                if y_scores:
                    logger.info(f"{task_name} Mean Pred Score: {np.mean(y_scores):.4f}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
