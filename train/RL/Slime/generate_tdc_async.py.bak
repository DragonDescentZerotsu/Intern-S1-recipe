"""
TDC Agent Generate Function for Slime RL Training

This module provides the main async generate function for training TDC classification
agents using the Slime framework. It implements a multi-turn agent loop with tool calling,
mirroring the logic from test_tdc_via_api_F1.py but adapted for Slime's training pipeline.

Usage in shell script:
    --custom-generate-function-path generate_tdc_async.generate
    --rollout-function-path generate_tdc_async.generate_rollout_fully_async
"""

import asyncio
import atexit
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from contextlib import nullcontext
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from slime.rollout.sglang_rollout import (
    GenerateState,
    generate_and_rm_group,
    generate_rollout as sglang_generate_rollout,
)
from slime.utils.async_utils import run
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Set up logger
logger = logging.getLogger(__name__)

# --- Tool infrastructure ---
# Add the project root to sys.path so we can import tools/utils
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools import BASIC_TOOLS, get_function_by_name
from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP

# --- Optional tracing infrastructure ---
_ENABLE_LANGFUSE = os.getenv("TDC_ENABLE_LANGFUSE", "0").strip().lower() in {"1", "true", "yes", "on"}
_LANGFUSE_MAX_TEXT_CHARS = int(os.getenv("TDC_LANGFUSE_MAX_TEXT_CHARS", "20000"))

try:
    from langfuse import get_client as _get_langfuse_client  # type: ignore
except Exception:
    _get_langfuse_client = None


def _truncate_text(text: str | None, limit: int = _LANGFUSE_MAX_TEXT_CHARS) -> str | None:
    if text is None or len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _resolve_langfuse_enabled(args) -> bool:
    """
    Resolve whether Langfuse should be enabled.
    Priority:
    1) args.enable_langfuse / args.langfuse (if present)
    2) TDC_ENABLE_LANGFUSE env var
    """
    if hasattr(args, "enable_langfuse"):
        return bool(getattr(args, "enable_langfuse"))
    if hasattr(args, "langfuse"):
        return bool(getattr(args, "langfuse"))
    return _ENABLE_LANGFUSE


def _get_langfuse_client_if_enabled(enable_langfuse: bool):
    if not enable_langfuse:
        return None
    if _get_langfuse_client is None:
        logger.warning("Langfuse is enabled but not installed. Skipping trace logging.")
        return None
    try:
        return _get_langfuse_client()
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse client: {e}")
        return None


def _log_langfuse_turn(
    langfuse_client,
    *,
    model_name: str,
    turn: int,
    task_name: str,
    messages_for_trace: list[dict],
    assistant_message: dict,
    tool_results: list[dict] | None,
    finish_reason: dict | None,
):
    if langfuse_client is None:
        return

    try:
        sanitized_messages = []
        for msg in messages_for_trace[-8:]:
            sanitized = {"role": msg.get("role", "unknown")}
            if "name" in msg:
                sanitized["name"] = msg.get("name")
            if "tool_call_id" in msg:
                sanitized["tool_call_id"] = msg.get("tool_call_id")
            if "content" in msg:
                sanitized["content"] = _truncate_text(str(msg.get("content", "")))
            if msg.get("reasoning_content") is not None:
                sanitized["reasoning_content"] = _truncate_text(str(msg.get("reasoning_content", "")))
            if msg.get("raw_content") is not None:
                sanitized["raw_content"] = _truncate_text(str(msg.get("raw_content", "")))
            if "tool_calls" in msg:
                sanitized["tool_calls"] = msg.get("tool_calls", [])
            sanitized_messages.append(sanitized)

        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name=f"tdc_turn_{turn}",
        ) as generation:
            generation.update(
                model=model_name or "unknown",
                input=sanitized_messages,
                output={
                    "content": _truncate_text(assistant_message.get("content")),
                    "reasoning_content": _truncate_text(assistant_message.get("reasoning_content")),
                    "raw_content": _truncate_text(assistant_message.get("raw_content")),
                    "tool_calls": assistant_message.get("tool_calls", []),
                    "tool_results": tool_results or [],
                },
                metadata={
                    "task_name": task_name,
                    "turn": turn,
                    "finish_reason": finish_reason or {},
                },
            )
    except Exception as e:
        logger.debug(f"Langfuse generation logging failed at turn {turn}: {e}")


_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_reasoning_content(text: str) -> tuple[str, str | None]:
    """
    Split assistant output into (content, reasoning_content).
    Supports standard <think>...</think> and partial '</think>' style outputs.
    """
    if not text:
        return "", None

    reasoning_chunks = [chunk.strip() for chunk in _THINK_BLOCK_PATTERN.findall(text) if chunk.strip()]
    if reasoning_chunks:
        content = _THINK_BLOCK_PATTERN.sub("", text).strip()
        reasoning_content = "\n\n".join(reasoning_chunks)
        return content, reasoning_content

    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        reasoning_content = head.replace("<think>", "").strip() or None
        content = tail.strip()
        return content, reasoning_content

    return text.strip(), None


def get_tools_for_task(task_name: str) -> list[dict]:
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    return BASIC_TOOLS + specific_tools


def parse_tool_calls(response: str, tools_info: list[dict], parser_type: str = "glm47") -> dict:
    """
    Parse tool calls from LLM response using sglang FunctionCallParser.
    """
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    from sglang.srt.managers.io_struct import Function, Tool

    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools_info
    ]
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser_type)
    normal_text, calls = parser.parse_non_stream(response)
    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls],
    }


def execute_tool_calls(calls: list[dict]) -> list[dict]:
    """Execute tool calls locally using the project tool implementations."""
    results = []
    for i, call in enumerate(calls):
        tool_name = call.get("name", "")
        try:
            raw_params = call.get("parameters", "{}")
            params = json.loads(raw_params) if isinstance(raw_params, str) else raw_params
            func = get_function_by_name(tool_name)
            if func is None:
                result_str = f"Error: Unknown tool '{tool_name}'"
            else:
                result = func(**params)
                result_str = str(result)
        except Exception as e:
            result_str = f"Error executing tool '{tool_name}': {e}"

        results.append({
            "name": tool_name,
            "content": result_str,
            "tool_call_id": f"call_{i}_{tool_name}",
        })
    return results


def _get_token_delta(
    tokenizer, messages: list[dict], tools_info: list[dict] = None
) -> tuple[list[int], list[int]]:
    """
    Calculate token delta for multi-turn conversations.
    Adapted from tau-bench trainable_agents.py.
    """
    curr = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False, tools=tools_info, enable_thinking=True, clear_thinking=False
    )

    if messages[-1]["role"] == "assistant":
        prev = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False, tools=tools_info, enable_thinking=True, clear_thinking=False
        )
        new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
        return new_tokens, [1] * len(new_tokens)
    else:
        prev = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=False, tokenize=False, tools=tools_info, enable_thinking=True, clear_thinking=False
        )
        new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
        return new_tokens, [0] * len(new_tokens)


async def generate(args, sample: Sample, sampling_params: dict) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory for TDC tasks.

    Multi-turn agent loop:
    1. Model generates a response (potentially with tool calls)
    2. Tool calls are parsed and executed locally
    3. Tool results are appended to conversation history
    4. Repeat until no more tool calls or depth limit reached
    """
    assert not args.partial_rollout, "Partial rollout is not supported for TDC agent interactions."

    # Initialize state
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # Extract task info from metadata
    metadata = sample.metadata or {}
    task_name = metadata.get("task_name", "")

    # Get tools for this task
    tools_info = get_tools_for_task(task_name)

    # Build initial messages:
    # - messages_for_model: exact conversation content used for token accounting and next-turn generation
    # - messages_for_trace: structured conversation for debugging/inspection
    messages_for_model = [{"role": "user", "content": sample.prompt}]
    messages_for_trace = [{"role": "user", "content": sample.prompt}]

    # Prepare initial prompt tokens
    prompt_text = state.tokenizer.apply_chat_template(
        messages_for_model,
        tokenize=False,
        add_generation_prompt=True,
        tools=tools_info,
        enable_thinking=True,
        clear_thinking=False,
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Tracking variables
    response_token_ids = []
    loss_masks = []
    depth_limit = 40
    last_tool_names = set()
    final_content = ""
    status = Sample.Status.COMPLETED
    num_tool_calls = 0
    turn_count = 0

    # Optional Langfuse tracing
    enable_langfuse = _resolve_langfuse_enabled(args)
    langfuse_client = _get_langfuse_client_if_enabled(enable_langfuse)
    model_name = getattr(args, "model", None) or getattr(args, "hf_checkpoint", None) or ""

    trace_metadata = {
        "task_name": task_name,
        "sample_index": sample.index,
        "label": sample.label,
        "enable_langfuse": bool(langfuse_client),
    }
    trace_cm = nullcontext()
    if langfuse_client is not None:
        try:
            trace_cm = langfuse_client.start_as_current_trace(
                name="slime_tdc_generate",
                input={"prompt": _truncate_text(str(sample.prompt))},
                metadata=trace_metadata,
            )
        except Exception as e:
            logger.debug(f"Langfuse trace init failed: {e}")
            trace_cm = nullcontext()

    with trace_cm:
        for turn in range(depth_limit):
            turn_count = turn + 1

            # Prepare text input for sglang
            text_input = state.tokenizer.apply_chat_template(
                messages_for_model,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools_info,
                enable_thinking=True,
                clear_thinking=False,
            )
            payload = {"text": text_input, "sampling_params": sampling_params}

            # Call sglang server
            try:
                output = await post(url, payload)
            except Exception as e:
                logger.warning(f"sglang request failed at turn {turn}: {e}")
                status = Sample.Status.ABORTED
                break

            # Check for abort
            finish_reason = output.get("meta_info", {}).get("finish_reason", {})
            if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                status = Sample.Status.ABORTED
                break

            response = output["text"].strip()

            # Remove end-of-turn tokens if present (model-specific)
            eos_tokens_to_strip = ["<|user|>", "<|endoftext|>", '<|observation|>']
            for eos_tok in eos_tokens_to_strip:
                if response.endswith(eos_tok):
                    response = response[:-len(eos_tok)]

            # Try to parse tool calls from the response
            tool_calls = []
            normal_text = response
            try:
                parsed = parse_tool_calls(response, tools_info)
                normal_text = parsed["normal_text"]
                tool_calls = parsed["calls"]
            except Exception as e:
                logger.debug(f"Tool call parsing returned no tools or failed: {e}")
                tool_calls = []

            response_content, reasoning_content = split_reasoning_content(normal_text)
            assistant_message_trace = {
                "role": "assistant",
                "content": response_content,
                "reasoning_content": reasoning_content,
                "raw_content": response,
                "tool_calls": tool_calls,
            }

            # Add assistant response to model messages (raw response preserved for exact token accounting)
            messages_for_model.append({"role": "assistant", "content": response})
            messages_for_trace.append(assistant_message_trace)

            assistant_tokens, assistant_mask = _get_token_delta(state.tokenizer, messages_for_model, tools_info)
            response_token_ids.extend(assistant_tokens)
            loss_masks.extend(assistant_mask)

            final_content = response_content

            # If no tool calls, we're done
            if not tool_calls:
                _log_langfuse_turn(
                    langfuse_client,
                    model_name=model_name,
                    turn=turn_count,
                    task_name=task_name,
                    messages_for_trace=messages_for_trace,
                    assistant_message=assistant_message_trace,
                    tool_results=[],
                    finish_reason=finish_reason if isinstance(finish_reason, dict) else {},
                )
                break

            # Check for repeated tool calls (prevent infinite loops, same as test script)
            current_tool_names = set(call.get("name", "") for call in tool_calls)
            if current_tool_names == last_tool_names:
                _log_langfuse_turn(
                    langfuse_client,
                    model_name=model_name,
                    turn=turn_count,
                    task_name=task_name,
                    messages_for_trace=messages_for_trace,
                    assistant_message=assistant_message_trace,
                    tool_results=[],
                    finish_reason=finish_reason if isinstance(finish_reason, dict) else {},
                )
                break
            last_tool_names = current_tool_names

            # Execute tool calls
            tool_results = execute_tool_calls(tool_calls)
            num_tool_calls += len(tool_results)

            # Add tool results to messages
            for tool_result in tool_results:
                messages_for_model.append(
                    {
                        "role": "tool",
                        "name": tool_result["name"],
                        "content": tool_result["content"],
                    }
                )
                messages_for_trace.append(
                    {
                        "role": "tool",
                        "name": tool_result["name"],
                        "content": tool_result["content"],
                        "tool_call_id": tool_result["tool_call_id"],
                    }
                )
                tool_tokens, tool_mask = _get_token_delta(state.tokenizer, messages_for_model, tools_info)
                response_token_ids.extend(tool_tokens)
                loss_masks.extend(tool_mask)

            _log_langfuse_turn(
                langfuse_client,
                model_name=model_name,
                turn=turn_count,
                task_name=task_name,
                messages_for_trace=messages_for_trace,
                assistant_message=assistant_message_trace,
                tool_results=tool_results,
                finish_reason=finish_reason if isinstance(finish_reason, dict) else {},
            )
        else:
            status = Sample.Status.TRUNCATED

    # Build final Sample
    all_token_ids = prompt_token_ids + response_token_ids
    full_response = ""
    full_raw_response = ""
    for msg in messages_for_trace:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            full_response += msg.get("content", "")
            full_raw_response += msg.get("raw_content", "")

    sample.tokens = all_token_ids
    sample.response = full_response
    sample.response_content = full_response
    sample.raw_response = full_raw_response
    sample.final_content = final_content
    sample.loss_mask = loss_masks
    sample.status = status
    sample.response_length = len(loss_masks)
    sample.metadata = {
        **(sample.metadata or {}),
        "num_turns": turn_count,
        "num_tool_calls": num_tool_calls,
        "task_name": task_name,
        "trajectory_messages": messages_for_trace,
    }

    if langfuse_client is not None:
        try:
            langfuse_client.update_current_trace(
                output={
                    "response_content": _truncate_text(full_response),
                    "raw_response": _truncate_text(full_raw_response),
                },
                metadata={
                    "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
                    "num_turns": turn_count,
                    "num_tool_calls": num_tool_calls,
                },
            )
        except Exception as e:
            logger.debug(f"Langfuse trace update failed: {e}")

    logger.info(
        f"TDC generate finished: task={task_name}, turns={turn_count}, "
        f"status={sample.status.value if hasattr(sample.status, 'value') else status}, "
        f"response_length={sample.response_length}"
    )
    return sample


# --- Fully async rollout infrastructure (adapted from slime/examples/fully_async) ---
_global_worker = None
_worker_lock = threading.Lock()


class AsyncRolloutWorker:
    """Background rollout worker that continuously pulls prompts and runs generation."""

    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)
        self.worker_thread = None
        self.state = GenerateState(args)

    async def continuous_worker_loop(self):
        logger.info("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = max(1, self.args.rollout_batch_size)
        group_id_counter = 0

        while self.running:
            try:
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()
                        except Exception as e:
                            logger.warning(f"Async rollout task failed: {e}")
                    active_tasks -= done_tasks

                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)
                    if not samples:
                        break

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        def task_done_callback(done_task, gid=group_id):
                            try:
                                result = done_task.result()
                            except Exception as e:
                                logger.warning(f"Async rollout callback failed for group {gid}: {e}")
                                return
                            try:
                                # [BUG FIX]: Use non-blocking `put_nowait` instead of blocking `put()`.
                                # [BUG FIX]: 使用非阻塞的 `put_nowait` 替代阻塞的 `put()`。
                                # Why: If the Trainer is busy doing Evaluation, it stops pulling items from this output_queue.
                                # If the queue reaching its maxsize (1000) using blocking `put()`, the asyncio worker thread 
                                # will be indefinitely suspended, leading to pipeline deadlock. 
                                # 为什么：如果主训练器正在花费大量时间做评测（Evaluation），它会停止从此 output_queue 获取数据。
                                # 此时若后台满载，阻塞式的 `put()` 会将整个 asyncio 工作线程无限期挂起（死锁）。使用 nowait 进行丢弃。
                                self.output_queue.put_nowait((gid, result))
                            except queue.Full:
                                logger.error(f"Worker queue is full! Dropping generated group {gid}.")

                        task.add_done_callback(task_done_callback)
                        active_tasks.add(task)

                        if len(active_tasks) >= max_concurrent_tasks:
                            break

                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(f"Waiting for {len(active_tasks)} async tasks to finish before shutdown")
            await asyncio.wait(active_tasks)

        logger.info("Continuous async rollout worker stopped")

    def start(self):
        from slime.utils.async_utils import get_async_loop
        if self.worker_thread is None or self.worker_thread.done():
            # [BUG FIX]: Route the worker to the Global Shared Event Loop rather than spawning a new native thread.
            # [BUG FIX]: 将后台 Worker 投递到“全局共享事件循环”中，而不是使用原生 threading.Thread 新起一个 Event Loop。
            # Why: SGLang's `GenerateState` object contains ans `asyncio.Semaphore` bounded to the first Event Loop that initialized it.
            # If Evaluation mode is natively triggered on the main thread, cross-loop calls will crash with:
            # `RuntimeError: Task got Future attached to a different loop`. This shared loop guarantees thread-safety.
            # 为什么：SGLang 的 `GenerateState` 中的协程信号量 `asyncio.Semaphore` 是单例且绑定到首个 Event Loop 上的。
            # 原本的独立新线程会导致主线程如果发起 Evaluation，两者事件循环发生跨域冲突直接崩溃（如上报错）。使用共享的全局异步循环解决冲突。
            self.worker_thread = asyncio.run_coroutine_threadsafe(
                self.continuous_worker_loop(),
                get_async_loop().loop
            )
            logger.info("Started continuous async worker task in AsyncLoopThread")

    def stop(self):
        self.running = False
        if self.worker_thread and not self.worker_thread.done():
            self.worker_thread.cancel()
        logger.info("Stopped async worker task")

    def get_completed_groups(self) -> list[tuple]:
        completed = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        return self.output_queue.qsize()


def get_global_worker(args, data_buffer):
    """Get or create a singleton background worker for rollout generation."""
    global _global_worker
    with _worker_lock:
        if (
            _global_worker is None
            or _global_worker.worker_thread is None
            or _global_worker.worker_thread.done()
        ):
            logger.info("Creating new global async rollout worker")
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global async worker if it exists."""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


async def generate_rollout_async(args, rollout_id: int, data_buffer) -> list[list[Sample]]:
    """Collect rollout groups from a persistent background async worker."""
    del rollout_id
    assert args.rollout_global_dataset

    worker = get_global_worker(args, data_buffer)
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True

    logger.info(f"Starting fully async rollout generation for {target_data_size} groups")
    logger.info(f"Initial worker queue size: {worker.get_queue_size()}")

    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        made_progress = False
        for group_id, group in completed:
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        processed_any = False
        for group_id in sorted(completed_groups.keys()):
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)
            try:
                any_aborted = any(sample.status == Sample.Status.ABORTED for sample in group)
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    data_buffer.add_samples([group])
                    logger.info(f"Returned aborted group {group_id} to data buffer")
                except Exception as e:
                    logger.warning(f"Failed to return aborted group {group_id} to buffer: {e}")
                continue

            if do_print and group:
                logger.info(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}"
                )
                do_print = False

            data.append(group)
            processed_any = True

        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            logger.warning(
                f"No rollout progress for {no_progress_timeout}s. "
                f"Queue size={worker.get_queue_size()}, collected={len(data)}/{target_data_size}"
            )
            last_progress_time = current_time

        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    logger.info(f"Fully async rollout completed in {duration:.2f}s, queue size={worker.get_queue_size()}")

    if data:
        logger.info(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}"
        )

    data = sorted(data, key=lambda group: group[0].index)
    return data


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation=False):
    """
    Fully async rollout entrypoint.

    For eval, fallback to Slime default rollout implementation so eval config
    can keep working even when training rollout is fully async.
    """
    if evaluation:
        return sglang_generate_rollout(args, rollout_id, data_buffer, evaluation=True)
    
    data = run(generate_rollout_async(args, rollout_id, data_buffer))
    
    from slime.rollout.base_types import RolloutFnTrainOutput
    
    metrics = {}
    if data:
        total_turns = 0
        total_tool_calls = 0
        total_samples = 0
        for group in data:
            for sample in group:
                total_samples += 1
                metadata = sample.metadata or {}
                total_turns += metadata.get("num_turns", 0)
                total_tool_calls += metadata.get("num_tool_calls", 0)
        
        if total_samples > 0:
            metrics["rollout/num_turns"] = total_turns / total_samples
            metrics["rollout/num_tool_calls"] = total_tool_calls / total_samples
            
    return RolloutFnTrainOutput(samples=data, metrics=metrics)


atexit.register(stop_global_worker)
