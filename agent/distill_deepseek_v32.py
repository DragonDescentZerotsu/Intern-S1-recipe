import sys
import os
import multiprocessing
import json
import time
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools import BASIC_TOOLS, get_function_by_name
# Import the tools map directly from the RDKit file since it's not exported by tools/__init__.py (based on my read) -- wait, checks tools/__init__.py again.
# tools/__init__.py imports * from RDKit_tools, so it should be available if RDKit_tools exports it.
# Let's check RDKit_tools.py again. Yes, TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP is defined there.
from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP

from utils.TDC_answer_parser import extract_answer, parse_answer
from utils.norm_messages_to_Alpaca import normalize_DeepSeek_V32_messages, split_into_steps
from utils.save_jsonl import save_jsonl
from transformers import AutoTokenizer



# Configs
Current_dir = Path(__file__).parent.resolve()
Project_root = Current_dir.parent

API_BASE = "http://0.0.0.0:8000/v1"
API_KEY = "EMPTY"
MAX_SAMPLES = 3
BATCH_SIZE = 128  # Rows per task chunk
OUTPUT_BASE_DIR = Project_root / "DataPrepare/SFT_data/DeepSeek_V32_distill_agent_data"
ALPACA_DIR = OUTPUT_BASE_DIR / "intern-s1-mini_TDC_train_Alpaca_per_task"  # 注意 Alpaca 格式需要自己指定，这份文件里面已经指定适应了 intern-s1 的格式
RAW_NORMALIZED_DIR = OUTPUT_BASE_DIR / "TDC_train_raw_normalized_messages_per_task"
STATE_FILE = Current_dir / "distill_state.json"
DATA_SOURCE_DIR = Project_root / "DataPrepare/TDC_train_prompts_label"

# Initialize directories
ALPACA_DIR.mkdir(parents=True, exist_ok=True)
RAW_NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

def get_tools_for_task(task_name: str) -> List[Dict]:
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    # Filter out potential duplicates if any, though lists are usually distinct in design
    # Actually BASIC_TOOLS is a list of tool definitions (dicts).
    # We can just concatenate them.
    return BASIC_TOOLS + specific_tools

def run_turn(client: OpenAI, messages: List[Dict], tools: List[Dict]) -> str:
    """Run a single conversation turn with the model."""
    model_name = client.models.list().data[0].id
    sub_turn = 1
    depth_limit = 30
    
    final_content = ""
    last_tool_names = set()
    
    while sub_turn <= depth_limit:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                extra_body={"chat_template_kwargs": {"thinking": True}}  # 这里 thinking 一定是开的
            )
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return None

        message = response.choices[0].message
        messages.append(message)
        
        tool_calls = message.tool_calls
        final_content = message.content or ""

        if not tool_calls:
            break
            
        current_tool_names = set(tool_call.function.name for tool_call in tool_calls)
        if current_tool_names == last_tool_names:
            # Infinite loop detection
            break
        last_tool_names = current_tool_names
            
        for tool_call in tool_calls:
            try:
                # We need to find the function implementation. 
                # tools/__init__.py has get_function_by_name which maps names to functions.
                # Note: The tool definitions in RDKit_tools.py map string names to python functions via the tool_map in tools/__init__.py
                func = get_function_by_name(tool_call.function.name)
                if func:
                    tool_call_args = json.loads(tool_call.function.arguments)
                    tool_call_result = func(**tool_call_args)
                    tool_call_result_str = str(tool_call_result)
                else:
                    tool_call_result_str = f"Error: Tool {tool_call.function.name} not found."
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


# We need to make sure that the pool processes are not daemonic so they can spawn children (AccFG tool does this)
# -------------------------------------------------------------------------
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
# -------------------------------------------------------------------------

def worker_process_sample(args):
    """
    Worker function to process a single prompt with rejection sampling.
    Args: (index, text, label, tools_def)
    """
    index, text, label, tools_def = args
    
    # Initialize client per process
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    
    # Try up to MAX_SAMPLES times
    for attempt_idx in range(MAX_SAMPLES):
        if attempt_idx > 0:
            print(f"Sample {index} (Labels: {label}): Retry {attempt_idx + 1}/{MAX_SAMPLES}...", flush=True)
        messages = [{'role': 'user', 'content': text}]
        try:
            response_text = run_turn(client, messages, tools_def)
            if response_text:
                ans_txt, fmt_ok = extract_answer(response_text)
                # Assuming think is enabled via chat template config
                prediction = parse_answer(ans_txt, fmt_ok, think_is_on=True)
                
                # Check if prediction matches label
                # label in dataset is usually 0 or 1
                if prediction is not None and prediction == label:
                    # Success!
                    return (index, messages)
        except Exception as e:
            # print(f"Error in sample loop: {e}")
            pass
            
    # If we reach here, we failed to get a correct answer in MAX_SAMPLES
    return (index, None)

class TaskScheduler:
    def __init__(self):
        self.state = self.load_state()
        self.tasks = self.discover_tasks()
        
    def discover_tasks(self) -> List[str]:
        """
        Discover tasks from the data source directory.
        从数据源目录中发现任务名字 task_name
        """
        files = list(DATA_SOURCE_DIR.glob("*.jsonl"))
        tasks = []
        for f in files:
            task_name = f.stem
            if task_name in ["SAbDab_Chen", "MHC1_IEDB-IMGT_Nielsen", "MHC2_IEDB_Jensen", "HuRI"]:  # 暂时不做的一些 tasks
                continue
            tasks.append(task_name)
        tasks.sort()
        return tasks
        
    def load_state(self) -> Dict:
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def save_state(self):
        # Update timestamp or similar if needed, but the important part is 'offset' and 'completed'
        with open(STATE_FILE, 'w') as f:
            json.dump(self.state, f, indent=2)
            
    def get_task_progress(self, task_name: str) -> Dict:
        if task_name not in self.state:
            self.state[task_name] = {"offset": 0, "completed": False}
        return self.state[task_name]
        
    def update_task_progress(self, task_name: str, new_offset: int, completed: bool):
        self.state[task_name] = {"offset": new_offset, "completed": completed}
        self.save_state()

def load_data_chunk(task_name: str, offset: int, limit: int) -> Tuple[List[Dict], bool]:
    """Load a chunk of data from the jsonl file."""
    file_path = DATA_SOURCE_DIR / f"{task_name}.jsonl"
    data = []
    
    # Read all first (not efficient for huge files, but these seem manageable ~MBs to GBs)
    # Alternatively, seek/tell could be used, but jsonl lines are variable length.
    # We can use islice.
    import itertools
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip 'offset' lines
        # Check if we can read enough
        try:
            # Attempt to read limit + 1 to check if finished
            # But simpler: read everything? No, "ToxCast.jsonl" is 1.3GB. better iterate.
            iterator = itertools.islice(f, offset, offset + limit)
            for line in iterator:
                if line.strip():
                    data.append(json.loads(line))
        except StopIteration:
            pass
            
    # Check if we reached end of file
    # We can try to read one more byte or line to see if we are done?
    # Or just rely on returned data length < limit (imperfect if exact multiple)
    # Let's check total line count only if needed, or just assume if len(data) < limit it's done.
    # To be safer:
    is_completed = False
    if len(data) < limit:
        is_completed = True
    else:
        # Check if there is more data
        with open(file_path, 'r', encoding='utf-8') as f:
            # wastefully checking... better way:
             # Just assume false, if next round returns 0, then mark done.
             pass
             
    # Optimization: If we got 0 data, it's definitely completed.
    if len(data) == 0:
        is_completed = True
        
    return data, is_completed

def main():
    scheduler = TaskScheduler()
    tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)
    
    print(f"Found {len(scheduler.tasks)} tasks.")
    
    # We loop until all tasks are marked completed
    while True:
        all_completed = True
        tasks_processed_this_round = 0
        
        for task_name in scheduler.tasks:
            progress = scheduler.get_task_progress(task_name)
            if progress["completed"]:
                continue
                
            all_completed = False
            tasks_processed_this_round += 1
            
            current_offset = progress["offset"]
            print(f"\nProcessing Task: {task_name}, Offset: {current_offset}")
            
            # Load Data
            chunk_data, is_possibly_done = load_data_chunk(task_name, current_offset, BATCH_SIZE)
            
            if not chunk_data:
                # No more data
                print(f"Task {task_name} finished!")
                scheduler.update_task_progress(task_name, current_offset, True)
                continue
                
            # Prepare args for multiprocessing
            tools_def = get_tools_for_task(task_name)
            
            # (index, text, label, tools_def)
            # Use actual index from file? offset + i
            work_items = []
            for i, item in enumerate(chunk_data):
                # We need text and Y (label)
                if 'text' in item and 'Y' in item:
                    work_items.append((current_offset + i, item['text'], item['Y'], tools_def))
            
            if not work_items:
                # Chunk empty or invalid?
                 scheduler.update_task_progress(task_name, current_offset + len(chunk_data), is_possibly_done)
                 continue

            # Run Batch
            results_to_save_raw = []
            results_to_save_alpaca = []
            
            # Using multiprocessing
            # num_processes = min(multiprocessing.cpu_count(), 32) # or fixed small number if API constrained
            num_processes = 32 # Conservative start
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                for res in tqdm(pool.imap_unordered(worker_process_sample, work_items), total=len(work_items), desc=f"{task_name} Batch"):
                    idx, messages = res
                    if messages: # Success
                         # Normalize
                         normalized_msgs = normalize_DeepSeek_V32_messages(messages)
                         
                         # Save Raw
                         # We can just save the list of messages as one line (standard format usually {messages: [...]})
                         # But user requested "save normalized messages as one line in a .jsonl file"
                         # So we wrap it: {"messages": normalized_msgs, "task": task_name, "index": idx}
                         # Or just the messages list?
                         # "normalize messages please refer to @[utils/norm_messages_to_Alpaca.py] ... save this normalized messages as one line in a .jsonl file"
                         # Usually this implies {"messages": ...} or just the list? The schema usually expects an object.
                         # Let's save {"messages": normalized_msgs, "task": task_name, "id": idx}
                         results_to_save_raw.append({"messages": normalized_msgs, "task": task_name, "origin_idx": idx})
                         
                         # Save Alpaca
                         # split_into_steps returns a list of items
                         alpaca_items = split_into_steps(normalized_msgs, tokenizer, tools=tools_def)
                         results_to_save_alpaca.extend(alpaca_items)
            
            # Write to disk
            if results_to_save_raw:
                raw_path = RAW_NORMALIZED_DIR / f"{task_name}_raw.jsonl"
                # Append mode
                with open(raw_path, 'a', encoding='utf-8') as f:
                    for r in results_to_save_raw:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        
            if results_to_save_alpaca:
                alpaca_path = ALPACA_DIR / f"{task_name}_alpaca.jsonl"
                with open(alpaca_path, 'a', encoding='utf-8') as f:
                    for r in results_to_save_alpaca:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
            # Update state
            new_offset = current_offset + len(chunk_data)
            scheduler.update_task_progress(task_name, new_offset, is_possibly_done)
            
        if all_completed:
            print("All tasks completed!")
            break
            
        if tasks_processed_this_round == 0:
             # Should be covered by all_completed, but safety break
             break

if __name__ == "__main__":
    # Safety for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
