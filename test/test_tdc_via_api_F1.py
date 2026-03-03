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
import re
import jinja2
from sklearn.metrics import f1_score, classification_report
from openai import OpenAI  # use this when don't need langfuse
# import logfire
# from langfuse.openai import OpenAI  # langfuse
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
_RUN_TURN = None

def get_tools_for_task(task_name):
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    # Filter out potential duplicates if any, though lists are usually distinct in design
    return BASIC_TOOLS + specific_tools

def get_args():
    parser = argparse.ArgumentParser(description='Run TDC benchmark tasks via OpenAI-compatible API (F1 Score)')
    
    parser.add_argument('--task-groups', nargs='+', default=['Tox', 'ADME', 'HTS', 'Develop'],  # TODO: test tasks
                        choices=['ADME', 'Tox', 'HTS', 'Develop', 'PPI', 'TCREpitopeBinding', 'TrialOutcome', 'PeptideMHC', 'Other', 'all'],
                        help='Task groups to run')
    parser.add_argument('--n-samples', type=int, default=1, help='Number of samples per query')  # sample only once for F1 score
    parser.add_argument('--api-base', type=str, default="http://localhost:8000/v1", help='API Base URL')  # TODO: port number | local model: http://localhost:8000/v1 | openrouter: https://openrouter.ai/api/v1 ｜ deepseek: https://api.deepseek.com/v1
    parser.add_argument('--api-key', type=str, default="EMPTY", help='API Key')  # TODO: API Key | local model: "EMPTY" | openrouter: os.environ["OPENROUTER_API_KEY_Haydn"], os.environ["OPENROUTER_API_KEY_Mark"] | deepseek: os.environ["DEEPSEEK_API_KEY"]
    parser.add_argument('--model', type=str, default="", help='Model name (optional, will query server if empty)')  # TODO: model name | local model: "" | openrouter: deepseek/deepseek-v3.2; openai/gpt-5.2; openai/gpt-5-mini | deepseek: deepseek-chat
    parser.add_argument('--num-processes', type=int, default=1, help='Number of parallel workers')  # 16 for intern-s1-mini, 8 for GLM-4.7-Flash
    parser.add_argument('--data-dir', type=Path, default=current_dir.parent / "DataPrepare/TDC_valid_prompts_label_scaffold", help='Directory containing processed test data')
    parser.add_argument('--thinking', action='store_true', default=True, help='Enable thinking parameter for DeepSeek models')  # TODO: 注意这里 thinking 到底是开了还是没开
    parser.add_argument('--enable-tools', action='store_true', default=True, help='Enable tool calling')  # TODO: 注意是否使用了 tool ， debug 可能关了
    parser.add_argument('--log-file', action='store_true', default=False, help='Save logs to file')  # TODO: 注意这里 log-file 到底是开了还是没开
    parser.add_argument('--log-file-name', type=str, default="Tools_gpt-oss-120b_{t_stamp}_dev_4.log", help='logs file name')   # TODO: log file name
    parser.add_argument('--langfuse', action='store_true', default=True, help='Save traces to langfuse')  # TODO: 注意这里 langfuse trace 到底是开了还是没开
    parser.add_argument('--max-retry', type=int, default=4, help='Max retries for answer parsing failure')
    parser.add_argument('--use-playbook', action='store_true', default=True, help='Inject playbook into the prompt')
    parser.add_argument('--score-based', action='store_true', default=True, help='If true, expect and parse a 0-100 probability instead of (A)/(B)')
    
    args = parser.parse_args()
    return args

def init_worker(api_base, api_key, use_langfuse, task_name):
    global _CLIENT, _RUN_TURN
    _CLIENT = OpenAI(
        api_key=api_key,
        base_url=api_base,
        # 让问题暴露得更明显，避免一直重试掩盖根因
        max_retries=1,          # 或 1/2
        timeout=60.0           # 视你模型速度调整
    )

    _RUN_TURN = get_run_turn(use_langfuse, task_name)

    if use_langfuse:
        # 每个 worker 进程退出时把队列刷出去
        atexit.register(lambda: get_client().flush())  # langfuse

def _get_usage_details(
    resp: Any,
    *,
    pricing: Optional[Dict[str, Any]] = None,  # OpenRouter "pricing" object for this model (optional)
) -> Tuple[Optional[Dict[str, int]], Optional[Dict[str, float]]]:
    """
    Returns:
      usage_details: Dict[str, int]  (Langfuse: units per usage type)
      cost_details:  Dict[str, float] (Langfuse: USD cost per usage type)

    Works best with OpenRouter usage accounting enabled:
      extra_body={"usage": {"include": True}}
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None

    def g(x: Any, k: str, default=None):
        if x is None:
            return default
        if isinstance(x, dict):
            return x.get(k, default)
        return getattr(x, k, default)

    # --- token counts ---
    prompt_tokens = g(usage, "prompt_tokens")
    completion_tokens = g(usage, "completion_tokens")
    total_tokens = g(usage, "total_tokens")

    # details dicts (provider-dependent)
    prompt_details = g(usage, "prompt_tokens_details", {}) or {}
    completion_details = g(usage, "completion_tokens_details", {}) or {}

    cached_tokens = g(prompt_details, "cached_tokens")  # OpenRouter uses cached_tokens :contentReference[oaicite:4]{index=4}
    reasoning_tokens = g(completion_details, "reasoning_tokens")  # OpenRouter example shows reasoning_tokens :contentReference[oaicite:5]{index=5}

    usage_details: Dict[str, int] = {}
    if prompt_tokens is not None:
        usage_details["input"] = int(prompt_tokens)
    if completion_tokens is not None:
        usage_details["output"] = int(completion_tokens)
    if cached_tokens is not None:
        usage_details["input_cache_read"] = int(cached_tokens)
    if reasoning_tokens is not None:
        # OpenRouter pricing object has "internal_reasoning" pricing field :contentReference[oaicite:6]{index=6}
        usage_details["internal_reasoning"] = int(reasoning_tokens)
    if total_tokens is not None:
        usage_details["total"] = int(total_tokens)

    # If pricing includes request cost, treat as 1 unit request
    if pricing and float(pricing.get("request", "0") or 0) != 0:
        usage_details["request"] = 1

    # --- cost ---
    cost_details: Dict[str, float] = {}

    # 1) Prefer provider-reported total cost if present (OpenRouter returns cost in credits; credits are USD-denominated) :contentReference[oaicite:7]{index=7}
    reported_cost = g(usage, "cost")
    if reported_cost is not None:
        try:
            cost_details["total"] = float(reported_cost)
        except Exception:
            pass

    # 2) If we have per-unit pricing, compute breakdown in USD (OpenRouter pricing is USD per token/request/unit) :contentReference[oaicite:8]{index=8}
    if pricing:
        def f(k: str) -> float:
            v = pricing.get(k, "0")
            try:
                return float(v)
            except Exception:
                return 0.0

        pt = int(prompt_tokens or 0)
        ct = int(completion_tokens or 0)
        cache = int(cached_tokens or 0)
        rt = int(reasoning_tokens or 0)

        input_cost = pt * f("prompt")
        output_cost = ct * f("completion")
        cache_read_cost = cache * f("input_cache_read")
        reasoning_cost = rt * f("internal_reasoning")
        request_cost = (1 * f("request")) if ("request" in usage_details) else 0.0

        # Only set keys if non-zero (avoid clutter)
        if input_cost:
            cost_details["input"] = input_cost
        if output_cost:
            cost_details["output"] = output_cost
        if cache_read_cost:
            cost_details["input_cache_read"] = cache_read_cost
        if reasoning_cost:
            cost_details["internal_reasoning"] = reasoning_cost
        if request_cost:
            cost_details["request"] = request_cost

        # If provider didn't give total, derive it
        if "total" not in cost_details:
            cost_details["total"] = float(
                input_cost + output_cost + cache_read_cost + reasoning_cost + request_cost
            )

    # If both are empty, return None to avoid writing junk into Langfuse
    if not usage_details:
        usage_details_out = None
    else:
        usage_details_out = usage_details

    if not cost_details:
        cost_details_out = None
    else:
        cost_details_out = cost_details

    return usage_details_out, cost_details_out

def get_run_turn(use_langfuse: bool, task_name: str):
    '''
    Returns the run_turn function with langfuse observation if use_langfuse is True.
    Otherwise, returns the run_turn_base function.
    '''
    if use_langfuse:
        return observe(
            as_type="agent",
            name=task_name,
            # capture_input=False,
            # capture_output=False,
        )(run_turn_base)
    return run_turn_base

# @observe(as_type="agent")  # langfuse
def run_turn_base(client, messages, model_name, thinking=False, tools=None, use_langfuse=False, ground_truth_label=None):
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
            # response = client.chat.completions.create(
            #     # name='repaired_QED',  # langfuse
            #     model=model_name,
            #     messages=messages,
            #     tools=tools,
            #     max_tokens=10240, # Reduced from 20000 to be safe/faster, usually enough. Important to keep this small other wise retry and slow down the speed.
            #     temperature=0.8,
            #     top_p=0.8,
            #     stream=False,
            #     extra_body=dict(spaces_between_special_tokens=False, enable_thinking=True)
            # )
            # TODO: local GLM-4.7-Flash
            # response = client.chat.completions.create(
            #     # name='repaired_QED',  # langfuse
            #     model=model_name,
            #     messages=messages,
            #     tools=tools,
            #     max_tokens=10240, # Reduced from 20000 to be safe/faster, usually enough. Important to keep this small other wise retry and slow down the speed.
            #     temperature=0.8,
            #     top_p=0.8,
            #     stream=False,
            #     extra_body={
            #         "spaces_between_special_tokens": False,
            #         "chat_template_kwargs": {
            #             "enable_thinking": True,
            #             "clear_thinking": False
            #         }
            #     }
            # )
            # TODO: local gpt-oss
            response = client.chat.completions.create(
                # name='repaired_QED',  # langfuse
                model=model_name,
                messages=messages,
                tools=tools,
                max_tokens=10240, # Reduced from 20000 to be safe/faster, usually enough. Important to keep this small other wise retry and slow down the speed.
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
            # TODO: OpenRouter
            # response = client.chat.completions.create(
            #     model=model_name,
            #     messages=messages,
            #     tools=tools,
            #     extra_body={ 
            #         "reasoning": {   # 使用 OpenAI SDK 的 reasoning 功能
            #             # "effort": "high",  # TODO: Turn this OFF for OpenRouter offered DeepSeek models
            #             # "summary": "auto"  # TODO: Turn this OFF for OpenRouter offered DeepSeek models
            #             "enabled": True  # TODO: Turn this ON for OpenRouter offered DeepSeek models
            #         },
            #         "usage": {"include": True},
            #         **({  # 在用 DeepSeek 并且要做 tool call 的时候指定模型提供商，别的时候暂时不指定
            #             "provider":{
            #                 "only":["DeepSeek"],
            #                 "allow_fallbacks": False  # 严格只使用 DeepSeek 更便宜
            #             }
            #         } if ('deepseek' in model_name and tools is not None) else {})
            #     },  
            # )
            # TODO: OpenAI official API (not implemented yet)

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return None

        message = response.choices[0].message
        messages.append(message)

        # langfuse
        if use_langfuse:
            langfuse = get_client()
            usage_details, cost_details = _get_usage_details(response, pricing=None)
            # reasoning = getattr(message, "reasoning_content", None)

            # Update trace metadata with ground truth label for easy verification
            if ground_truth_label is not None:
                langfuse.update_current_trace(
                    metadata={
                        "ground_truth_label": ground_truth_label,
                        "expected_answer": "(B)" if ground_truth_label == 1 else "(A)",
                    }
                )

            with langfuse.start_as_current_observation(as_type="generation", name=model_name) as gen:
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
        # if reasoning:
            # 注意：reasoning 可能很长，Langfuse 有事件大小限制，太大会被丢弃；建议截断保存 :contentReference[oaicite:2]{index=2}
            # reasoning_trunc = reasoning[:20000]  # 你可以调大/调小

            # 把它放到当前 span 的 output（UI里更直观），再加点统计信息到 metadata
            # langfuse.update_current_span(
            #     output={
            #         "final_content": message.content,
            #         "reasoning_content": reasoning_trunc,
            #     },
            #     metadata={
            #         "has_reasoning_content": True,
            #         "reasoning_chars": len(reasoning),
            #         "reasoning_truncated_to": len(reasoning_trunc),
            #     },
            # )

        if not tool_calls:
            break
            
        # Check if the set of tool names is the same as the previous turn to prevent infinite loops
        current_tool_names = set(tool_call.function.name for tool_call in tool_calls)
        # if current_tool_names == last_tool_names:
        #     break
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

def load_playbook(task_name):
    project_root = Path(__file__).resolve().parent.parent
    playbook_file = f"{task_name}.jinja2"
    
    modified_path = project_root / "playbooks" / "modified" / playbook_file
    origin_path = project_root / "playbooks" / "Haydn_origin" / playbook_file
    
    if modified_path.exists():
        target_path = modified_path
    elif origin_path.exists():
        target_path = origin_path
    else:
        return ""
        
    class PlaybookLoader(jinja2.BaseLoader):
        def get_source(self, environment, template):
            if template == "tdc/base.jinja2":
                path = project_root / "playbooks" / "modified" / "base.jinja2"
                if not path.exists():
                    path = project_root / "playbooks" / "Haydn_origin" / "base.jinja2"
            else:
                path = project_root / "playbooks" / "modified" / template
                if not path.exists():
                    path = project_root / "playbooks" / "Haydn_origin" / template
            
            if not path.exists():
                raise jinja2.TemplateNotFound(template)
            with open(path, 'r', encoding='utf-8') as f:
                source = f.read()
            return source, str(path), lambda: True

    env = jinja2.Environment(loader=PlaybookLoader())
    template = env.get_template(playbook_file)
    return template.render()

def worker_process_sample(args):
    """
    Args: (index, text, label_dummy, api_base, api_key, model_name, thinking, enable_tools, task_name, use_langfuse, max_retry, use_playbook, score_based)
    Returns: (index, prediction_int_or_None)
    """
    global _CLIENT, _RUN_TURN

    index, text, label, api_base, api_key, model_name, thinking, enable_tools, task_name, use_langfuse, max_retry, use_playbook, score_based = args
    ground_truth_label = int(label)  # Keep track of ground truth for Langfuse metadata
    
    # client = OpenAI(
    #     api_key=api_key,
    #     base_url=api_base,
    # )
    client = _CLIENT

    # Determine tools
    tools = None
    if enable_tools:
        tools = get_tools_for_task(task_name)
    
    if score_based:
        # Change prompt text to require 0-100 instead of A/B
        if "Drug SMILES:" in text:
            text = text.split("Drug SMILES:")[1].split("\n")[0].strip()
        user_text = f'The given SMILES is {text}, follow the above instructions and think step by step to make your prediction. Output a probability from 0 to 100 after "Answer: " (e.g. "Answer: 50").'
    
    if use_playbook:
        playbook_text = load_playbook(task_name)
        # if playbook_text:
        #     text = playbook_text + "\n\n" + text

    messages = [
        {'role': 'system', 'content': playbook_text},
        {'role': 'user', 'content': text}
        ]

    for attempt in range(max_retry):
        try:
            # We need to make a copy of messages because _RUN_TURN might append to it (though in current impl it appends, but if we retry we want fresh start? 
            # Actually _RUN_TURN appends to the passed list. If we retry, we should reset messages to original state or just pass a fresh copy.
            # unique messages for each attempt to avoid accumulating history from failed attempts if that's desired?
            # Usually for retry we want a fresh start.
            current_messages = [m.copy() for m in messages]
            
            response_text, tool_count = _RUN_TURN(client, current_messages, model_name, thinking, tools, use_langfuse, ground_truth_label)
            if response_text:
                # parsing logic
                # Ensure extract_answer and parse_answer are available
                from utils.TDC_answer_parser import extract_answer, parse_answer
                
                ans_txt, fmt_ok = extract_answer(response_text)
                prediction = parse_answer(ans_txt, fmt_ok, think_is_on=thinking, score_based=score_based) # Assuming thinking affects parsing logic?
                # Note: parse_answer signature might vary. 
                # In skin-reaction.py: parse_answer(ans_txt, fmt_ok, think_is_on=True)
                
                if prediction is not None:
                    return (index, prediction, tool_count)
        except Exception as e:
            # logger.error(f"Worker error attempt {attempt}: {e}")
            pass
        
    return (index, None, 0)


def load_tasks_map(data_dir):
    """
    Maps task groups to specific files in the data directory.
    This mapping relies on the file naming convention from process_tdc_test.py.
    """
    mapping = {  # TODO: Detailed Task selection
        'Tox': [
            # 'hERG_Karim.jsonl',  # 2690
            # 'Carcinogens_Lagunin.jsonl',  # 56
            # 'Skin_Reaction.jsonl',  # 82
            # 'hERG.jsonl',  # 132
            # 'DILI.jsonl',  # 96
            # 'ClinTox.jsonl',  # 297
            # 'AMES.jsonl',  # 1457
            # 'Tox21.jsonl',  # 15584
            # -----------------------------------------
            # 'herg_central_hERG_inhib.jsonl',  # 61379    leave out
            # 'ToxCast.jsonl'  # 307282    leave out
        ],
        'ADME': [
            # 'PAMPA_NCATS.jsonl',  # 408
            # 'HIA_Hou.jsonl',  # 117
            'BBB_Martins.jsonl',  # 406
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
            # 'HIV.jsonl',  # 8225
            # 'SARSCoV2_3CLPro_Diamond.jsonl',  # 176
            # 'SARSCoV2_Vitro_Touret.jsonl',  # 298
            # -----------------------------------------
            # 'butkiewicz.jsonl'  # 401997    leave out
        ],
        # 'Develop': ['SAbDab_Chen.jsonl'],  # 482
        # 'PPI': ['HuRI.jsonl'],  # 20282
        # 'TCREpitopeBinding': ['Weber.jsonl'],  # not in our test set yet
        'TrialOutcome': [
            # 'phase1.jsonl', 
            # 'phase2.jsonl', 
            # 'phase3.jsonl'
            ],
        'PeptideMHC': [
            # 'MHC1_IEDB-IMGT_Nielsen.jsonl',  # 37197
            # 'MHC2_IEDB_Jensen.jsonl'  # 26856
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
        log_path = log_dir / args.log_file_name.format(t_stamp=timestamp)
        
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)
        logger.info(f"Logging to file: {log_path}")

    logger.info(f"Running groups: {groups_to_run}")

    all_results = {} # group -> task -> score

    for group in groups_to_run:
        if group not in task_map:
            logger.warning(f"No files found for group {group} or group not defined.")
            continue
            
        files = task_map[group]
        all_results[group] = {}
        
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
            worker_tasks = []
            for idx, item in enumerate(raw_data):
                task_tuple = (idx, item['text'], item['Y'], args.api_base, args.api_key, args.model, args.thinking, args.enable_tools, task_name, args.langfuse, args.max_retry, args.use_playbook, args.score_based)
                for _ in range(args.n_samples):
                    worker_tasks.append(task_tuple)
            
            # Execute
            results = []
            with multiprocessing.Pool(
                processes=args.num_processes,
                initializer=init_worker,
                initargs=(args.api_base, args.api_key, args.langfuse, task_name)
            ) as pool:
                for result in tqdm(pool.imap_unordered(worker_process_sample, worker_tasks), total=len(worker_tasks), desc=f"{task_name}"):
                    results.append(result)
            
            # Aggregate
            item_results = {} # idx -> list of predictions
            item_tool_counts = {} # idx -> list of tool counts
            
            # --- Failure Statistics (None predictions) ---
            total_samples_count = len(results)
            none_pred_count = sum(1 for _, p, _ in results if p is None)
            if total_samples_count > 0:
                none_rate = (none_pred_count / total_samples_count) * 100
                logger.info(f"{task_name} Failure Rate (pred is None): {none_pred_count}/{total_samples_count} ({none_rate:.2f}%)")
            
            for idx, pred, tool_count in results:
                if idx not in item_results:
                    item_results[idx] = []
                    item_tool_counts[idx] = []
                if pred is not None:
                    item_results[idx].append(pred)
                    item_tool_counts[idx].append(tool_count)
            
            # Calculate metrics (F1)
            y_true = []
            y_pred = []
            failed_parses_count = 0
            
            for idx in range(len(raw_data)):
                label = int(raw_data[idx]['Y'])
                y_true.append(label)
                
                preds = item_results.get(idx, [])
                # Filter None (already done essentially, but double check)
                valid_preds = [p for p in preds if p is not None]

                if not valid_preds:
                    failed_parses_count += 1
                    # Forces wrong prediction for F1 punishment
                    y_pred.append(1 - label)
                else:
                    # Majority Voting
                    count_1 = valid_preds.count(1)
                    count_0 = valid_preds.count(0)
                    
                    if count_1 >= count_0:
                        final_pred = 1
                    else:
                        final_pred = 0
                    y_pred.append(final_pred)

            
            # Final Report for this Task
            logger.info("\n" + "=" * 80)
            logger.info(f"{task_name} EVALUATION RESULTS (F1 Score)")
            logger.info("=" * 80)
            logger.info(f"Failed parses (all samples failed for a q): {failed_parses_count}/{len(raw_data)}")
            
            if len(set(y_true)) > 1:
                score = f1_score(y_true, y_pred, average='macro', pos_label=1)
                logger.info(f"Classification Report:\n{classification_report(y_true, y_pred, digits=4, labels=[0, 1])}")
                logger.info(f"F1 Score: {score:.4f}")
                all_results[group][task_name] = score
            else:
                logger.info(f"{task_name} F1: Cannot calculate (one class only in truth)")
                if y_pred:
                    logger.info(f"Mean Pred: {np.mean(y_pred):.4f}")
                all_results[group][task_name] = 0.0

            logger.info("=" * 80 + "\n")

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

    if args.langfuse:
        get_client().flush()  # langfuse
    
    # ===================== Summary =====================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY RESULTS (F1 Score)")
    logger.info("=" * 80)
    
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
    multiprocessing.set_start_method('spawn', force=True)
    main()
