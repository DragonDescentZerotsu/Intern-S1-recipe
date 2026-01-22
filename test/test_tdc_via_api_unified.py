"""
Unified TDC evaluation via OpenAI-compatible API with tool calling.

Combines:
- Tool calling logic from test_tdc_via_api_AUROC.py (multiprocessing, run_turn_base)
- Results saving structure from eval_vllm.py (results/eval/{task}/{config}/...)

Example usage:
    # Local vLLM server
    python test_tdc_via_api_unified.py --task-groups ADME --api-base http://localhost:8000/v1 --model Intern-S1-mini --specialized-model intern --enable-tools

    # OpenRouter
    python test_tdc_via_api_unified.py --task-groups Tox --model deepseek/deepseek-v3.2 --enable-tools

    # Test mode (limit samples)
    python test_tdc_via_api_unified.py --task-groups ADME --test --test-samples 2
"""

import os
import argparse
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import numpy as np
from sklearn.metrics import roc_auc_score
from openai import OpenAI
from langfuse import observe, get_client
import atexit
from dotenv import load_dotenv
load_dotenv()

# Add project root to path for imports
import sys
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.parent  # therapeutic-tuning/
sys.path.append(str(project_root))
sys.path.append(str(current_dir.parent))  # Intern-S1-recipe/

from tools import BASIC_TOOLS, get_function_by_name

# Verify these imports exist in your project structure
try:
    from tools import * 
    from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP
    from utils.TDC_answer_parser import extract_answer, parse_answer
except ImportError:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Global client instance
_CLIENT = None
_RUN_TURN = None

def get_tools_for_task(task_name):
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    return BASIC_TOOLS + specific_tools

def get_args():
    parser = argparse.ArgumentParser(description='Run TDC benchmark tasks via OpenAI-compatible API')
    
    # Task selection
    parser.add_argument('--task-groups', nargs='+', default=['Tox', 'ADME', 'HTS'],
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'Other', 'all'],
                        help='Task groups to run')
    
    # API configuration
    parser.add_argument('--api-base', type=str, default="https://openrouter.ai/api/v1",
                        help='API Base URL (local: http://localhost:8000/v1)')
    parser.add_argument('--api-key', type=str, default=os.environ.get("OPENROUTER_API_KEY_Mark_1", "EMPTY"),
                        help='API Key (use EMPTY for local vLLM)')
    parser.add_argument('--model', type=str, default="deepseek/deepseek-v3.2",
                        help='Model name')
    
    # Evaluation parameters
    parser.add_argument('--n-samples', type=int, default=8, help='Number of samples per query')
    parser.add_argument('--num-processes', type=int, default=256, help='Number of parallel workers')
    parser.add_argument('--data-dir', type=Path, default=current_dir.parent / "DataPrepare/TDC_test_prompts_label_scaffold",
                        help='Directory containing processed test data')
    
    # Feature flags
    parser.add_argument('--thinking', action='store_true', default=True, help='Enable thinking/reasoning')
    parser.add_argument('--no-thinking', action='store_true', help='Disable thinking/reasoning')
    parser.add_argument('--enable-tools', action='store_true', default=True, help='Enable tool calling')
    parser.add_argument('--no-tools', action='store_true', help='Disable tool calling')
    parser.add_argument('--langfuse', action='store_true', help='Enable Langfuse tracing')
    
    # Test mode (from eval_vllm.py)
    parser.add_argument('--test', action='store_true',
                        help='Test mode: use "test" as date folder AND limit samples')
    parser.add_argument('--test-samples', type=int, default=2,
                        help='Number of samples per task in test mode')
    
    # Output configuration (from eval_vllm.py)
    parser.add_argument('--specialized-model', type=str, default=None,
                        help='Model type for folder naming (e.g., intern, deepseek)')
    parser.add_argument('--augmented-tools', type=str, default="TaskSpecific",
                        help='Tool config for folder naming (e.g., TaskSpecific, all, none)')
    parser.add_argument('--augmented-features', type=str, default=None,
                        help='Feature config for folder naming')
    
    # Legacy logging (from original)
    parser.add_argument('--log-file', action='store_true', help='Also save logs to file')
    parser.add_argument('--log-file-name', type=str, default="api_eval_{t_stamp}.log", help='Log file name')
    
    args = parser.parse_args()
    
    # Handle negation flags
    if args.no_thinking:
        args.thinking = False
    if args.no_tools:
        args.enable_tools = False
        
    return args

def init_worker(api_base, api_key, use_langfuse, task_name):
    global _CLIENT, _RUN_TURN
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=api_base,
        max_retries=1,
        timeout=180.0
    )

    _RUN_TURN = get_run_turn(use_langfuse, task_name)

    if use_langfuse:
        atexit.register(lambda: get_client().flush())

def _get_usage_details(
    resp: Any,
    *,
    pricing: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]]]:
    """Extract usage and cost details from API response."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None

    def g(x: Any, k: str, default=None):
        if x is None:
            return default
        if isinstance(x, dict):
            return x.get(k, default)
        return getattr(x, k, default)

    prompt_tokens = g(usage, "prompt_tokens")
    completion_tokens = g(usage, "completion_tokens")
    total_tokens = g(usage, "total_tokens")
    prompt_details = g(usage, "prompt_tokens_details", {}) or {}
    completion_details = g(usage, "completion_tokens_details", {}) or {}
    cached_tokens = g(prompt_details, "cached_tokens")
    reasoning_tokens = g(completion_details, "reasoning_tokens")

    usage_details: Dict[str, int] = {}
    if prompt_tokens is not None:
        usage_details["input"] = int(prompt_tokens)
    if completion_tokens is not None:
        usage_details["output"] = int(completion_tokens)
    if cached_tokens is not None:
        usage_details["input_cache_read"] = int(cached_tokens)
    if reasoning_tokens is not None:
        usage_details["internal_reasoning"] = int(reasoning_tokens)
    if total_tokens is not None:
        usage_details["total"] = int(total_tokens)

    if pricing and float(pricing.get("request", "0") or 0) != 0:
        usage_details["request"] = 1

    cost_details: Dict[str, float] = {}
    reported_cost = g(usage, "cost")
    if reported_cost is not None:
        try:
            cost_details["total"] = float(reported_cost)
        except Exception:
            pass

    return usage_details if usage_details else None, cost_details if cost_details else None

def get_run_turn(use_langfuse: bool, task_name: str):
    """Returns the run_turn function with optional langfuse observation."""
    if use_langfuse:
        return observe(as_type="agent", name=task_name)(run_turn_base)
    return run_turn_base

def run_turn_base(client, messages, model_name, thinking=False, tools=None, use_langfuse=False):
    """
    Executes a multi-turn conversation with optional tool support.
    """
    depth_limit = 40
    sub_turn = 1
    last_tool_names = set()
    final_content = ""

    extra_body = {}
    if thinking:
        extra_body = {"chat_template_kwargs": {"thinking": True}}

    while sub_turn <= depth_limit:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                extra_body={ 
                    "reasoning": {"enabled": True},
                    "usage": {"include": True},
                    **({
                        "provider": {
                            "only": ["DeepSeek"],
                            "allow_fallbacks": False
                        }
                    } if ('deepseek' in model_name.lower() and tools is not None) else {})
                },  
            )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None

        message = response.choices[0].message
        messages.append(message)

        if use_langfuse:
            langfuse = get_client()
            usage_details, cost_details = _get_usage_details(response, pricing=None)
            with langfuse.start_as_current_observation(as_type="generation", name="api-call") as gen:
                gen.update(
                    model=response.model,
                    input=[m.model_dump() if hasattr(m, "model_dump") else m for m in messages],
                    output={
                        "content": message.content,
                        "reasoning_content": getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None),
                    },
                    usage_details=usage_details,
                    cost_details=cost_details,
                )
        
        tool_calls = message.tool_calls
        final_content = message.content

        if not tool_calls:
            break
            
        current_tool_names = set(tool_call.function.name for tool_call in tool_calls)
        if current_tool_names == last_tool_names:
            break
        last_tool_names = current_tool_names
            
        for tool_call in tool_calls:
            try:
                tool_call_args = json.loads(tool_call.function.arguments)
                tool_call_result = get_function_by_name(tool_call.function.name)(**tool_call_args)
                tool_call_result_str = str(tool_call_result)
            except Exception as e:
                tool_call_result_str = f"Error executing tool: {e}"
            
            messages.append({
                'role': 'tool',
                'name': tool_call.function.name,
                'content': tool_call_result_str,
                'tool_call_id': tool_call.id
            })
        sub_turn += 1
        
    total_tool_calls = 0
    for msg in messages:
        if getattr(msg, 'role', '') == 'assistant' and getattr(msg, 'tool_calls', None):
            total_tool_calls += len(msg.tool_calls)
            
    return final_content, total_tool_calls

def worker_process_sample(args):
    """Process a single sample in a worker process."""
    global _CLIENT, _RUN_TURN

    index, text, _, api_base, api_key, model_name, thinking, enable_tools, task_name, use_langfuse = args
    
    client = _CLIENT
    messages = [{'role': 'user', 'content': text}]
    
    tools = None
    if enable_tools:
        tools = get_tools_for_task(task_name)
    
    try:
        response_text, tool_count = _RUN_TURN(client, messages, model_name, thinking, tools, use_langfuse)
        if response_text:
            from utils.TDC_answer_parser import extract_answer, parse_answer
            ans_txt, fmt_ok = extract_answer(response_text)
            prediction = parse_answer(ans_txt, fmt_ok, think_is_on=thinking)
            return (index, prediction, tool_count, response_text)
    except Exception as e:
        pass
        
    return (index, None, 0, None)

def load_tasks_map(data_dir):
    """Maps task groups to specific files in the data directory."""
    mapping = {
        'Tox': [
            'Skin_Reaction.jsonl',
            'hERG.jsonl',
            'DILI.jsonl',
            'ClinTox.jsonl',
            'AMES.jsonl',
        ],
        'ADME': [
            'PAMPA_NCATS.jsonl',
            'HIA_Hou.jsonl',
            'BBB_Martins.jsonl',
            'Pgp_Broccatelli.jsonl',
            'Bioavailability_Ma.jsonl',
            'CYP2C9_Substrate_CarbonMangels.jsonl',
            'CYP2D6_Substrate_CarbonMangels.jsonl',
            'CYP3A4_Substrate_CarbonMangels.jsonl'
        ],
        'HTS': [
            'SARSCoV2_3CLPro_Diamond.jsonl',
            'SARSCoV2_Vitro_Touret.jsonl',
        ],
        'Develop': ['SAbDab_Chen.jsonl'],
        'PPI': ['HuRI.jsonl'],
        'TrialOutcome': ['phase1.jsonl', 'phase2.jsonl', 'phase3.jsonl'],
        'PeptideMHC': ['MHC1_IEDB-IMGT_Nielsen.jsonl', 'MHC2_IEDB_Jensen.jsonl']
    }
    
    available_files = set(f.name for f in Path(data_dir).glob("*.jsonl"))
    
    final_map = {}
    for group, filenames in mapping.items():
        existing = [f for f in filenames if f in available_files]
        if existing:
            final_map[group] = existing
            
    return final_map


def get_output_dir(task_name: str, config_folder: str, date_str: str) -> Path:
    """Get output directory following eval_vllm.py structure."""
    return project_root / "results" / "eval" / task_name / config_folder / date_str


def main():
    args = get_args()
    
    data_path = args.data_dir
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

    # ==================== Output directory setup (from eval_vllm.py) ====================
    model_short = args.model.split('/')[-1]
    
    # Date string
    if args.test:
        date_str = "test"
    else:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Config folder name: {model}_{specialized}_{tools}_tools_{features}_features
    specialized_part = args.specialized_model if args.specialized_model else "base"
    tools_part = args.augmented_tools if args.augmented_tools else ("enabled" if args.enable_tools else "none")
    features_part = args.augmented_features if args.augmented_features else "none"
    
    config_folder = f"{model_short}_{specialized_part}_{tools_part}_tools_{features_part}_features"
    
    # Setup logging
    logs_dir = project_root / "results" / "eval" / "_logs" / config_folder / date_str
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    fh = logging.FileHandler(logs_dir / "eval.log")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(fh)
    
    logger.info(f"Model: {args.model}")
    logger.info(f"Config Folder: {config_folder}")
    logger.info(f"Date: {date_str}")
    logger.info(f"Thinking: {args.thinking}")
    logger.info(f"Tools: {args.enable_tools}")
    logger.info(f"Test Mode: {args.test}")

    task_map = load_tasks_map(data_path)
    
    groups_to_run = args.task_groups
    if 'all' in groups_to_run:
        groups_to_run = list(task_map.keys())

    logger.info(f"Running groups: {groups_to_run}")

    # Track all results for summary
    all_eval_results = {}

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group} or group not defined.")
            continue
            
        files = task_map[group]
        
        for filename in files:
            file_path = data_path / filename
            task_name = filename.replace('.jsonl', '')
            
            logger.info(f"Processing Task: {task_name}")
            
            # Create output directory for this task
            task_output_dir = get_output_dir(task_name, config_folder, date_str)
            task_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Output Dir: {task_output_dir}")
            
            # Load Data
            raw_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
            
            if not raw_data:
                logger.warning(f"No data in {filename}")
                continue

            # Test mode: limit samples
            if args.test and len(raw_data) > args.test_samples:
                logger.info(f"  Test mode: limiting to {args.test_samples} samples (was {len(raw_data)})")
                raw_data = raw_data[:args.test_samples]

            # Prepare args for multiprocessing
            worker_tasks = []
            for idx, item in enumerate(raw_data):
                task_tuple = (idx, item['text'], item['Y'], args.api_base, args.api_key, args.model, args.thinking, args.enable_tools, task_name, args.langfuse)
                for _ in range(args.n_samples):
                    worker_tasks.append(task_tuple)
            
            # Execute with multiprocessing
            results = []
            first_response_text = None
            with multiprocessing.Pool(
                processes=min(args.num_processes, len(worker_tasks)),
                initializer=init_worker,
                initargs=(args.api_base, args.api_key, args.langfuse, task_name)
            ) as pool:
                for result in tqdm(pool.imap_unordered(worker_process_sample, worker_tasks), total=len(worker_tasks), desc=f"{task_name}"):
                    results.append(result)
                    # Capture first response for sample output
                    if first_response_text is None and result[3] is not None:
                        first_response_text = result[3]
            
            # Aggregate results
            item_results = {}
            item_tool_counts = {}
            
            # Failure statistics
            total_samples_count = len(results)
            none_pred_count = sum(1 for _, p, _, _ in results if p is None)
            if total_samples_count > 0:
                none_rate = (none_pred_count / total_samples_count) * 100
                logger.info(f"{task_name} Failure Rate: {none_pred_count}/{total_samples_count} ({none_rate:.2f}%)")
            
            for idx, pred, tool_count, _ in results:
                if idx not in item_results:
                    item_results[idx] = []
                    item_tool_counts[idx] = []
                if pred is not None:
                    item_results[idx].append(pred)
                    item_tool_counts[idx].append(tool_count)
            
            # Calculate metrics
            y_true = []
            y_scores = []
            all_preds = []
            
            for idx in range(len(raw_data)):
                if idx not in item_results or not item_results[idx]:
                    all_preds.append(None)
                    continue
                
                preds = item_results[idx]
                count_0 = preds.count(0)
                count_1 = preds.count(1)
                total_valid = count_0 + count_1
                
                if total_valid == 0:
                    all_preds.append(None)
                    continue
                
                score = count_1 / total_valid
                y_true.append(int(raw_data[idx]['Y']))
                y_scores.append(score)
                all_preds.append(score)
            
            # Compute AUROC
            auroc = 0.0
            if len(set(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_scores)
                logger.info(f"{task_name} AUROC: {auroc:.4f}")
            else:
                logger.info(f"{task_name} AUROC: Cannot calculate (one class or empty)")
                if y_scores:
                    logger.info(f"{task_name} Mean Pred Score: {np.mean(y_scores):.4f}")

            # Tool usage statistics
            total_tool_calls = 0
            if args.enable_tools:
                all_tool_counts = []
                questions_with_tool_usage = 0
                total_questions = 0

                for idx in item_tool_counts:
                    counts = item_tool_counts[idx]
                    if not counts: 
                        continue
                    
                    all_tool_counts.extend(counts)
                    total_questions += 1
                    
                    if any(c > 0 for c in counts):
                        questions_with_tool_usage += 1
                
                total_tool_calls = sum(all_tool_counts)
                avg_tool_calls = np.mean(all_tool_counts) if all_tool_counts else 0.0
                usage_rate = (questions_with_tool_usage / total_questions) * 100 if total_questions > 0 else 0.0
                
                logger.info(f"{task_name} Avg Tools/Sample: {avg_tool_calls:.2f}")
                logger.info(f"{task_name} Questions w/ Tools: {questions_with_tool_usage}/{total_questions} ({usage_rate:.1f}%)")

            # ==================== Save results (from eval_vllm.py) ====================
            
            # eval_results.json
            valid_idx = [i for i, p in enumerate(all_preds) if p is not None]
            task_eval_results = {
                "test": {
                    "auroc": auroc,
                    "valid_samples": len(valid_idx),
                    "total_samples": len(raw_data),
                    "none_count": len(raw_data) - len(valid_idx),
                    "tool_calls_total": total_tool_calls,
                }
            }
            
            results_file = task_output_dir / "eval_results.json"
            with open(results_file, 'w') as f:
                json.dump(task_eval_results, f, indent=2)
            logger.info(f"  Results saved to {results_file}")
            
            # sample_outputs.json
            sample_outputs = [{
                "task": task_name,
                "split": "test",
                "prompt_full": raw_data[0]['text'] if raw_data else "",
                "label": raw_data[0]['Y'] if raw_data else None,
                "generation": first_response_text,
                "parsed_samples": item_results.get(0, []),
                "tool_calls": sum(item_tool_counts.get(0, [])),
                "tools_enabled": args.enable_tools,
            }]
            
            samples_file = task_output_dir / "sample_outputs.json"
            with open(samples_file, 'w') as f:
                json.dump(sample_outputs, f, indent=2, ensure_ascii=False)
            logger.info(f"  Sample outputs saved to {samples_file}")
            
            # Track for summary
            all_eval_results[task_name] = task_eval_results

    # ==================== Summary print (from eval_vllm.py) ====================
    if args.langfuse:
        get_client().flush()

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Config: {config_folder}")
    print(f"Tools Enabled: {args.enable_tools}")
    print("-"*60)
    for task, splits in all_eval_results.items():
        for split, metrics in splits.items():
            auroc_val = metrics.get('auroc', 0)
            valid = metrics.get('valid_samples', 0)
            total = metrics.get('total_samples', 0)
            print(f"{task}/{split}: AUROC = {auroc_val:.4f} ({valid}/{total} valid)")
    print("="*60)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
