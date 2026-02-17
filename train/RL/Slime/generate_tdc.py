"""
TDC Agent Generate Function for Slime RL Training

This module provides the main async generate function for training TDC classification
agents using the Slime framework. It implements a multi-turn agent loop with tool calling,
mirroring the logic from test_tdc_via_api_F1.py but adapted for Slime's training pipeline.

Usage in shell script:
    --custom-generate-function-path generate_tdc.generate
"""

import json
import logging
import os
import sys
from typing import Any

from slime.rollout.sglang_rollout import GenerateState
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


def get_tools_for_task(task_name: str) -> list[dict]:
    """Combine BASIC_TOOLS with task-specific tools."""
    specific_tools = TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])
    return BASIC_TOOLS + specific_tools


def parse_tool_calls(response: str, tools_info: list[dict], parser_type: str = "glm4") -> dict:
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
        messages, add_generation_prompt=False, tokenize=False, tools=tools_info
    )

    if messages[-1]["role"] == "assistant":
        prev = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False, tools=tools_info
        )
        new_tokens = tokenizer.encode(curr[len(prev):], add_special_tokens=False)
        return new_tokens, [1] * len(new_tokens)
    else:
        prev = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=False, tokenize=False, tools=tools_info
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

    # Build initial messages
    messages = [{"role": "user", "content": sample.prompt}]

    # Prepare initial prompt tokens
    prompt_text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, tools=tools_info
    )
    prompt_token_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Tracking variables
    response_token_ids = []
    loss_masks = []
    depth_limit = 40
    last_tool_names = set()
    final_content = ""
    status = "completed"

    for turn in range(depth_limit):
        # Prepare text input for sglang
        text_input = state.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, tools=tools_info
        )
        payload = {"text": text_input, "sampling_params": sampling_params}

        # Call sglang server
        try:
            output = await post(url, payload)
        except Exception as e:
            logger.warning(f"sglang request failed at turn {turn}: {e}")
            status = "aborted"
            break

        # Check for abort
        finish_reason = output.get("meta_info", {}).get("finish_reason", {})
        if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
            status = "aborted"
            break

        response = output["text"]

        # Remove end-of-turn tokens if present (model-specific)
        eos_tokens_to_strip = ["<|im_end|>", "<|endoftext|>"]
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
            # No tool calls - this is fine, model just gave a direct answer
            normal_text = response
            tool_calls = []

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response})
        assistant_tokens, assistant_mask = _get_token_delta(state.tokenizer, messages, tools_info)
        response_token_ids.extend(assistant_tokens)
        loss_masks.extend(assistant_mask)

        final_content = normal_text

        # If no tool calls, we're done
        if not tool_calls:
            break

        # Check for repeated tool calls (prevent infinite loops, same as test script)
        current_tool_names = set(call.get("name", "") for call in tool_calls)
        if current_tool_names == last_tool_names:
            break
        last_tool_names = current_tool_names

        # Execute tool calls
        tool_results = execute_tool_calls(tool_calls)

        # Add tool results to messages
        for tool_result in tool_results:
            messages.append({
                "role": "tool",
                "name": tool_result["name"],
                "content": tool_result["content"],
            })
            tool_tokens, tool_mask = _get_token_delta(state.tokenizer, messages, tools_info)
            response_token_ids.extend(tool_tokens)
            loss_masks.extend(tool_mask)

    # Build final Sample
    all_token_ids = prompt_token_ids + response_token_ids
    full_response = ""
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            full_response += msg.get("content", "")

    sample.tokens = all_token_ids
    sample.response = full_response
    sample.loss_mask = loss_masks
    sample.status = status
    sample.response_length = len(loss_masks)
    sample.metadata = {
        **(sample.metadata or {}),
        "num_turns": turn + 1,
        "num_tool_calls": sum(
            1 for msg in messages
            if isinstance(msg, dict) and msg.get("role") == "tool"
        ),
    }

    logger.info(
        f"TDC generate finished: task={task_name}, turns={turn + 1}, "
        f"status={status}, response_length={sample.response_length}"
    )
    return sample
