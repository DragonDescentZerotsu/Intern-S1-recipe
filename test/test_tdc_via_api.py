import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import multiprocessing
import numpy as np
from sklearn.metrics import roc_auc_score
# from openai import OpenAI  # use this when don't need langfuse
import logfire
from langfuse.openai import OpenAI  # langfuse
from langfuse import observe, get_client
import atexit
from dotenv import load_dotenv
load_dotenv()

# Add project root to path to import utils/tools if needed
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import BASIC_TOOLS, get_function_by_name

# Verify these imports exist in your project structure
try:
    from tools import * 
    from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP
    from utils.TDC_answer_parser import extract_answer, parse_answer
except ImportError:
    # Fallback if specific tools/utils are not easily importable or if running standalone
    # We will define a simple extractor if import fails, but prefer project utils
    pass

current_dir = Path(__file__).parent.resolve()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)  # 隐藏 HTTP 请求日志
# logging.getLogger("openai").setLevel(logging.DEBUG)
# logging.getLogger("openai._base_client").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Global client instance
_CLIENT = None

def get_tools_for_task(task_name):
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    # Filter out potential duplicates if any, though lists are usually distinct in design
    return BASIC_TOOLS + specific_tools

def get_args():
    parser = argparse.ArgumentParser(description='Run TDC benchmark tasks via OpenAI-compatible API')
    
    parser.add_argument('--task-groups', nargs='+', default=['ADME'],  # TODO: test tasks
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'Other', 'all'],
                        help='Task groups to run')
    parser.add_argument('--n-samples', type=int, default=16, help='Number of samples per query')
    parser.add_argument('--api-base', type=str, default="http://localhost:8000/v1", help='API Base URL')  # TODO: port number
    parser.add_argument('--api-key', type=str, default="EMPTY", help='API Key')
    parser.add_argument('--model', type=str, default="", help='Model name (optional, will query server if empty)')
    parser.add_argument('--num-processes', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--data-dir', type=Path, default=current_dir.parent / "DataPrepare/TDC_test_prompts_label", help='Directory containing processed test data')
    parser.add_argument('--thinking', action='store_false', help='Enable thinking parameter for DeepSeek models')  # TODO: 注意这里 thinking 到底是开了还是没开
    parser.add_argument('--enable-tools', action='store_false', help='Enable tool calling')  # TODO: 注意是否使用了 tool ， debug 可能关了
    parser.add_argument('--log-file', action='store_true', help='Save logs to file')  # TODO: 注意这里 log-file 到底是开了还是没开
    parser.add_argument('--logfire', action='store_false', help='Save logs to file')  # TODO: 注意这里 logfire 到底是开了还是没开
    
    args = parser.parse_args()
    return args

def init_worker(api_base, api_key):
    global _CLIENT
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=api_base,
        # 让问题暴露得更明显，避免一直重试掩盖根因
        max_retries=0,          # 或 1/2
        # timeout=120.0           # 视你模型速度调整
    )
    # 每个 worker 进程退出时把队列刷出去
    atexit.register(lambda: get_client().flush())  # langfuse

@observe()  # langfuse
def run_turn(client, messages, model_name, thinking=False, tools=None):
    """
    Executes a single turn of conversation with optional tool support.
    """
    depth_limit = 40 # Avoid infinite loops
    sub_turn = 1
    last_tool_names = set()
    final_content = ""

    extra_body = {}
    if thinking:
            extra_body = {"chat_template_kwargs": {"thinking": True}} # vLLM Server style

    while sub_turn <= depth_limit:
        try:
            # TODO: local Intern-S1-mini
            response = client.chat.completions.create(
                name='repaired_QED',
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=10240, # Reduced from 20000 to be safe/faster, usually enough. Important to keep this small other wise retry and slow down the speed.
                temperature=0.8,
                top_p=0.8,
                stream=False,
                extra_body=dict(spaces_between_special_tokens=False, enable_thinking=True)
            )
            # TODO: local DeepSeek V3.2
            # response = client.chat.completions.create(
            #     model=model_name,
            #     messages=messages,
            #     tools=tools,
            #     # max_tokens=30000,  # Reduced from 20000 to be safe/faster, usually enough
            #     # temperature=1.0,
            #     # top_p=0.95,
            #     # stream=False,
            #     extra_body={"chat_template_kwargs": {"thinking": True}} # vLLM Server
            # )
            # TODO: DeepSeek V3.2
            # response = client.chat.completions.create(
            #     model='deepseek-chat',
            #     messages=messages,
            #     tools=tools,
            #     extra_body={ "thinking": { "type": "enabled" } }  # 使用 OpenAI SDK 的 thinking 功能
            # )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None

        message = response.choices[0].message
        messages.append(message)
        
        tool_calls = message.tool_calls
        final_content = message.content

        if not tool_calls:
            break
            
        # Check if the set of tool names is the same as the previous turn to prevent infinite loops
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
        
    # Calculate total tool calls (excluding initial user message and final response)
    # We count 'tool_calls' in assistant messages.
    # Note: The simple way is to count how many times we found tool_calls in the loop. 
    # But since we didn't track it cumulatively, let's just count from the messages list or return a counter.
    # Let's count tool messages in the history as a proxy for successful tool executions? 
    # Or better, just count how many tool_call blocks we processed.
    # Let's iterate messages to be accurate on what happened.
    total_tool_calls = 0
    for msg in messages:
        if getattr(msg, 'role', '') == 'assistant' and getattr(msg, 'tool_calls', None):
            total_tool_calls += len(msg.tool_calls)
            
    return final_content, total_tool_calls

def worker_process_sample(args):
    """
    Args: (index, text, label_dummy, api_base, api_key, model_name, thinking, enable_tools, task_name)
    Returns: (index, prediction_int_or_None)
    """
    global _CLIENT

    index, text, _, api_base, api_key, model_name, thinking, enable_tools, task_name = args
    
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url=api_base,
    # )
    client = _CLIENT

    messages = [{'role': 'user', 'content': text}]
    
    # Determine tools
    tools = None
    if enable_tools:
        tools = get_tools_for_task(task_name)
    
    try:
        response_text, tool_count = run_turn(client, messages, model_name, thinking, tools)
        if response_text:
            # parsing logic
            # Ensure extract_answer and parse_answer are available
            from utils.TDC_answer_parser import extract_answer, parse_answer
            
            ans_txt, fmt_ok = extract_answer(response_text)
            prediction = parse_answer(ans_txt, fmt_ok, think_is_on=thinking) # Assuming thinking affects parsing logic?
            # Note: parse_answer signature might vary. 
            # In skin-reaction.py: parse_answer(ans_txt, fmt_ok, think_is_on=True)
            
            return (index, prediction, tool_count)
    except Exception as e:
        # logger.error(f"Worker error: {e}")
        pass
        
    return (index, None, 0)

def load_tasks_map(data_dir):
    """
    Maps task groups to specific files in the data directory.
    This mapping relies on the file naming convention from process_tdc_test.py.
    """
    mapping = {
        'Tox': [
            # --------------------------- Tox Group 1
            # 'Tox21.jsonl',  # 15589
            # 'ToxCast.jsonl',  # 306679
            # 'herg_central_hERG_inhib.jsonl'  # 61379
            # --------------------------- Tox Group 2
            'Skin_Reaction.jsonl',  # 81
            'hERG.jsonl',  # 131
            'DILI.jsonl',  # 95
            'ClinTox.jsonl',  # 296
            # --------------------------- Tox Group 3
            # 'AMES.jsonl',  # 1456
        ],
        'ADME': [
            # --------------------------- ADME Group 1
            'PAMPA_NCATS.jsonl',  # 407
            # 'HIA_Hou.jsonl',  # 116

            # 'BBB_Martins.jsonl',  # 406
            # 'Pgp_Broccatelli.jsonl',  # 244

            # 'Bioavailability_Ma.jsonl',  # 128
            # 'CYP2C9_Substrate_CarbonMangels.jsonl',  # 134
            # 'CYP2D6_Substrate_CarbonMangels.jsonl',  # 133
            # 'CYP3A4_Substrate_CarbonMangels.jsonl'  # 134
            # --------------------------- ADME Group 2
            # 'CYP1A2_Veith.jsonl',  # 2516
            # 'CYP2C19_Veith.jsonl',  # 2533
            # 'CYP2C9_Veith.jsonl',  # 2418
            # 'CYP2D6_Veith.jsonl',  # 2626
            # 'CYP3A4_Veith.jsonl',  # 2466
        ],
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

    # if args.logfire:
    #     logfire.configure()
    #     logfire.instrument_openai()
    
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

    task_map = load_tasks_map(data_path)
    
    groups_to_run = args.task_groups
    if 'all' in groups_to_run:
        groups_to_run = list(task_map.keys())

    # Configure File Handler if log_file is provided
    if args.log_file:
        log_dir = current_dir.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"intern_s1_mini_distilled_3000_steps_no_tools_{timestamp}_2.log"  # TODO: log file name
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

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
                task_tuple = (idx, item['text'], item['Y'], args.api_base, args.api_key, args.model, args.thinking, args.enable_tools, task_name)
                for _ in range(args.n_samples):
                    worker_tasks.append(task_tuple)
            
            # Execute
            results = []
            with multiprocessing.Pool(
                processes=args.num_processes,
                initializer=init_worker,
                initargs=(args.api_base, args.api_key)
            ) as pool:
                for result in tqdm(pool.imap_unordered(worker_process_sample, worker_tasks), total=len(worker_tasks), desc=f"{task_name}"):
                    results.append(result)
            
            # Aggregate
            item_results = {} # idx -> list of predictions
            item_tool_counts = {} # idx -> list of tool counts
            
            for idx, pred, tool_count in results:
                if idx not in item_results:
                    item_results[idx] = []
                    item_tool_counts[idx] = []
                if pred is not None:
                    item_results[idx].append(pred)
                    item_tool_counts[idx].append(tool_count)
            
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

            # --- Tool Usage Statistics ---
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
                    
                    # Check if at least one sample for this question used tools
                    if any(c > 0 for c in counts):
                        questions_with_tool_usage += 1
                
                avg_tool_calls = np.mean(all_tool_counts) if all_tool_counts else 0.0
                usage_rate = (questions_with_tool_usage / total_questions) * 100 if total_questions > 0 else 0.0
                
                logger.info(f"{task_name} Avg Tools/Sample: {avg_tool_calls:.2f}")
                logger.info(f"{task_name} Questions w/ Tools: {questions_with_tool_usage}/{total_questions} ({usage_rate:.1f}%)")
    get_client().flush()  # langfuse

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
