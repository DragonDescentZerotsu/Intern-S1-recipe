      
import sys
import os
import multiprocessing
import json
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from openai import OpenAI
import re
from transformers import AutoTokenizer, AutoProcessor
tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)  # for Debug use

from chat_templates.encoding_dsv32 import encode_messages, parse_message_from_completion_text  # for DeepSeek V3.2 Debug use
encode_config = dict(thinking_mode="thinking", drop_thinking=False, add_default_bos_token=True)
# usage: 
# prompt = encode_messages(messages, **encode_config)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import *
from utils.TDC_answer_parser import extract_answer, parse_answer

from pathlib import Path

Current_dir = Path(__file__).parent.resolve()

tools = TOOLS

def run_turn(client, messages):
    model_name = client.models.list().data[0].id
    sub_turn = 1
    depth_limit = 30 # Avoid infinite loops
    
    final_content = ""
    last_tool_names = set()
    
    while sub_turn <= depth_limit:
        try:
            # TODO: local Intern-S1-mini
            # response = client.chat.completions.create(
            #     model=model_name,
            #     messages=messages,
            #     tools=tools,
            #     max_tokens=20000, # Reduced from 20000 to be safe/faster, usually enough
            #     temperature=0.8,
            #     top_p=0.8,
            #     stream=False,
            #     extra_body=dict(spaces_between_special_tokens=False, enable_thinking=True)
            # )
            # TODO: local DeepSeek V3.2
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                # max_tokens=30000,  # Reduced from 20000 to be safe/faster, usually enough
                # temperature=1.0,
                # top_p=0.95,
                # stream=False,
                extra_body={"chat_template_kwargs": {"thinking": True}} # vLLM Server
            )
            # TODO: DeepSeek V3.2
            # response = client.chat.completions.create(
            #     model='deepseek-chat',
            #     messages=messages,
            #     tools=tools,
            #     extra_body={ "thinking": { "type": "enabled" } }  # 使用 OpenAI SDK 的 thinking 功能
            # )
        except Exception as e:
            print(f"Error in chat completion: {e}")
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
                tool_call_result_str = tool_call_result  # json.dumps(tool_call_result, ensure_ascii=False)
            except Exception as e:
                tool_call_result_str = f"Error executing tool: {e}"
            
            messages.append({
                'role': 'tool',
                'name': tool_call.function.name,
                'content': tool_call_result_str,
                'tool_call_id': tool_call.id
            })
        sub_turn += 1
        
    return final_content

def worker_process_sample(args):
    """
    Args: (index, text, label)
    Returns: (index, prediction_int_or_None)
    """
    index, text, label = args
    
    # Initialize client per process
    # TODO: local model
    # openai_api_key = "EMPTY"
    # openai_api_base = "http://0.0.0.0:23333/v1"
    # TODO: DeepSeek V3.2 GB200 local served
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:8000/v1"
    # TODO: DeepSeek V3.2 online API
    # openai_api_key=os.environ.get('DEEPSEEK_API_KEY')
    # openai_api_base='https://api.deepseek.com/v1'

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    messages = [{'role': 'user', 'content': text}]
    
    try:
        response_text = run_turn(client, messages)
        if response_text:
            ans_txt, fmt_ok = extract_answer(response_text)
            # parse_answer(answer_text, format_correct, think_is_on) - assuming think is enabled in request
            prediction = parse_answer(ans_txt, fmt_ok, think_is_on=True)
            return (index, prediction)
    except Exception:
        pass
        
    return (index, None)

def main():
    # Load data
    jsonl_path = Current_dir.parent / 'shared_data' / 'Skin_Reaction_test_vanilla.jsonl'
    
    print(f"Loading data from {jsonl_path}")
    
    raw_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    # Prepare tasks: 16 samples per item
    N_SAMPLES = 16
    tasks = []
    
    print(f"Total items: {len(raw_data)}. Generating {N_SAMPLES} samples per item. Total tasks: {len(raw_data) * N_SAMPLES}")
    
    for idx, item in enumerate(raw_data):
        # Tuple: (index, text, label)
        # We don't need label in worker, but we keep index to aggregate later
        task = (idx, item['text'], item['Y'])
        for _ in range(N_SAMPLES):
            tasks.append(task)
            
    # Run multiprocessing
    # Adjust processes based on server capacity. 
    # If vLLM is on one GPU, too many concurrent requests might increase latency.
    # 32 or 64 is usually okay for high throughput serving.
    num_processes = 1  # TODO: num_processes, change to 1 for debug
    
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        for result in tqdm(pool.imap_unordered(worker_process_sample, tasks), total=len(tasks)):
            results.append(result)
            
    # Aggregate results
    # item_results: index -> list of predictions (0, 1, or None)
    item_results = {}
    for idx, pred in results:
        if idx not in item_results:
            item_results[idx] = []
        if pred is not None:
            item_results[idx].append(pred)
            
    # Calculate scores and AUROC
    y_true = []
    y_scores = []
    
    success_count = 0
    
    for idx in range(len(raw_data)):
        if idx not in item_results or not item_results[idx]:
            # No valid predictions for this item
            continue
            
        preds = item_results[idx]
        count_0 = preds.count(0)
        count_1 = preds.count(1)
        total_valid = count_0 + count_1
        
        if total_valid == 0:
            continue
            
        # Score = P(class=1) = count(1) / total
        score = count_1 / total_valid
        
        y_true.append(raw_data[idx]['Y'])
        y_scores.append(score)
        success_count += 1
        
    print(f"Structure with valid predictions: {success_count}/{len(raw_data)}")
    
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_scores)
        print(f"\n{'='*40}")
        print(f"AUROC: {auroc:.4f}")
        print(f"{'='*40}\n")
    else:
        print("Cannot calculate AUROC: only one class present in valid results or no valid results.")

if __name__ == '__main__':
    # Set start method to spawn for safety with cuda/libraries
    multiprocessing.set_start_method('spawn', force=True)
    main()

