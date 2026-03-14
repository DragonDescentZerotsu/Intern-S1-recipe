import asyncio
import logging
import tinker
from config import cfg, MAX_CONTEXT_TOKENS, SAMPLE_TIMEOUT_SEC, STOP_TOKEN_IDS
from adapters import ModelAdapter
from reward import compute_reward

# We need to import the tool logic from the root `tools`
import sys
from config import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from tools import BASIC_TOOLS, get_function_by_name
from tools.RDKit_tools import TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP

logger = logging.getLogger(__name__)

# Used to track tokens for loss masking
TOKENS_STORE = []

def get_tools_for_task(task_name: str) -> list[dict]:
    return BASIC_TOOLS + TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.get(task_name, [])

def execute_tools(calls: list[dict]) -> list[dict]:
    import json
    results = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
        func = get_function_by_name(name)
        if func is None: content = f"Error: Unknown tool '{name}'"
        else:
            try: content = str(func(**args))
            except Exception as e: content = f"Error: {e}"
        results.append({"name": name, "content": content})
    return results

def build_messages(adapter: ModelAdapter, task_name: str, prompt_text: str) -> tuple[list[dict], list[dict]]:
    """Build messages and return (messages, tools) tuple."""
    if cfg.use_tools:
        tools = get_tools_for_task(task_name)
        sp = "You are a pharmaceutical research assistant with access to computational chemistry tools. Think step by step and use your tools to solve the problem."
        suffix = adapter.tools_as_system_suffix(tools)
        if suffix:
            sp += suffix
    else:
        tools = []
        sp = (
            "You are a pharmaceutical research assistant. "
            "Think step by step about the molecular properties and answer the question. "
            "When you have a final answer, state it clearly as: Answer: (A) or Answer: (B)"
        )
    # msgs = [{"role": "system", "content": sp}]
    msgs = []
    msgs.append({"role": "user", "content": prompt_text})
    return msgs, tools

def tok_messages(adapter: ModelAdapter, messages: list[dict], tools: list[dict]) -> list[int]:
    kwargs = adapter.apply_chat_template_kwargs(tools)
    text = adapter.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )
    return adapter.tokenizer.encode(text, add_special_tokens=False)

def _get_token_delta(adapter: ModelAdapter, messages: list[dict], tools: list[dict]) -> tuple[list[int], list[int]]:
    """
    Compute the new tokens added by the last message, with loss mask.
    Renders via string (where templates are reliable), then tokenizes
    both strings identically to diff in token space.
    """
    kwargs = adapter.apply_chat_template_kwargs(tools)
    is_assistant = messages[-1]["role"] == "assistant"

    curr_str = adapter.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
        **kwargs,
    )
    prev_str = adapter.tokenizer.apply_chat_template(
        messages[:-1],
        add_generation_prompt=is_assistant,
        tokenize=False,
        **kwargs,
    )
    if len(TOKENS_STORE) < 5:
      TOKENS_STORE.append({"curr_str": curr_str, "prev_str": prev_str})

    # Tokenize both strings through the same encode path
    curr_ids = adapter.tokenizer.encode(curr_str, add_special_tokens=False)
    prev_ids = adapter.tokenizer.encode(prev_str, add_special_tokens=False)

    # Find divergence point
    prefix_len = 0
    for i in range(min(len(prev_ids), len(curr_ids))):
        if prev_ids[i] != curr_ids[i]:
            break
        prefix_len = i + 1

    if prefix_len < len(prev_ids):
        # Template mutated earlier content — log details for debugging
        logger.warning(
            f"_get_token_delta: token prefix mismatch. "
            f"prev={len(prev_ids)} tok, curr={len(curr_ids)} tok, "
            f"prefix_len={prefix_len} (expected {len(prev_ids)}). "
            f"role={messages[-1]['role']}."
        )
        str_match = curr_str[:len(prev_str)] == prev_str
        if str_match:
            suffix_str = curr_str[len(prev_str):]
            new_tokens = adapter.tokenizer.encode(suffix_str, add_special_tokens=False)
            mask_val = 1 if is_assistant else 0
            return new_tokens, [mask_val] * len(new_tokens)
        else:
            logger.warning(
                f"  String prefix also mismatches. "
                f"prev_str ends: ...{repr(prev_str[-80:])}\n"
                f"  curr_str at same pos: ...{repr(curr_str[len(prev_str)-80:len(prev_str)+40])}"
            )

    new_tokens = curr_ids[prefix_len:]
    mask_val = 1 if is_assistant else 0
    return new_tokens, [mask_val] * len(new_tokens)


async def _single_rollout(adapter: ModelAdapter, sampling_client, task_name: str, prompt_text: str, gt: int, sparams):
    messages, tools = build_messages(adapter, task_name, prompt_text)
    full_text = ""
    last_tool_names = set()

    turn_data = []  # collect (prompt_tokens, gen_tokens, gen_logprobs) per turn

    for turn in range(cfg.max_turns):
        cur_tokens = tok_messages(adapter, messages, tools)

        if len(cur_tokens) + cfg.max_tokens > MAX_CONTEXT_TOKENS:
            break

        inp = tinker.types.ModelInput.from_ints(cur_tokens)
        fut = sampling_client.sample(prompt=inp, num_samples=1, sampling_params=sparams)
        try:
            res = await asyncio.to_thread(lambda: fut.result(timeout=SAMPLE_TIMEOUT_SEC))
        except Exception as e:
            logger.warning(f"Rollout sample failed: {e}")
            break
        seq = res.sequences[0]

        gen_tokens = list(seq.tokens)
        gen_lps = list(seq.logprobs) if seq.logprobs is not None else [0.0] * len(gen_tokens)
        if len(gen_lps) < len(gen_tokens):
            gen_lps = gen_lps + [0.0] * (len(gen_tokens) - len(gen_lps))
        gen_lps = gen_lps[:len(gen_tokens)]

        turn_data.append({
            "prompt_tokens": cur_tokens,
            "gen_tokens": gen_tokens,
            "gen_logprobs": gen_lps,
        })

        text = adapter.tokenizer.decode(gen_tokens, skip_special_tokens=adapter.skip_special_on_decode())
        for eos in adapter.eos_strips():
            if text.endswith(eos):
                text = text[:-len(eos)]
        full_text += text

        assistant_msg = adapter.format_assistant_message(text)
        messages.append(assistant_msg)

        calls = adapter.parse_tool_calls(text)
        if not calls:
            break
        
        # Check if the model is repeating the same tool calls
        # 检查模型是否在重复调用相同的工具，如果是就直接退出了，但是我觉得这里可能得改一下，暂时先去掉了，可以重复调用修正错误
        # current_names = set(c.get("name", "") for c in calls)
        # if current_names == last_tool_names:
        #     break
        # last_tool_names = current_names
        
        results = execute_tools(calls)
        for i, tr in enumerate(results):
            messages.append(adapter.format_tool_result_message(
                tr["name"], tr["content"], call_id=f"call_{i}"
            ))

    final_text = adapter.extract_final_text(full_text)
    reward = compute_reward(full_text, final_text, gt)

    return {
        "turn_data": turn_data,
        "reward": reward,
        "response_text": full_text,
        "messages": messages,
    }


async def run_batch_rollouts(adapter, sampling_client, batch, sparams):
    """Fire all rollouts for entire batch concurrently."""
    coros = []
    sample_indices = []
    for i, s in enumerate(batch):
        for _ in range(cfg.group_size):
            coros.append(_single_rollout(
                adapter, sampling_client, s["task_name"], s["text"], int(s["Y"]), sparams
            ))
            sample_indices.append(i)

    results = await asyncio.gather(*coros, return_exceptions=True)

    grouped = [[] for _ in batch]
    for idx, res in zip(sample_indices, results):
        if isinstance(res, Exception):
            logger.warning(f"Rollout failed: {res}")
            continue
        grouped[idx].append(res)

    return [(s, g) for s, g in zip(batch, grouped) if g]


def validate_rollout(adapter, rollout, sample):
    errors = []

    for i, td in enumerate(rollout["turn_data"]):
        p, g, lp = td["prompt_tokens"], td["gen_tokens"], td["gen_logprobs"]
        if len(g) != len(lp):
            errors.append(f"turn {i}: gen_tokens={len(g)} != gen_logprobs={len(lp)}")
        if len(g) == 0:
            errors.append(f"turn {i}: empty generation")
        bad_lps = sum(1 for x in lp if x > 0.0)
        if bad_lps:
            errors.append(f"turn {i}: {bad_lps} positive logprobs")

    has_any_logprobs = any(
        any(x != 0.0 for x in td["gen_logprobs"])
        for td in rollout["turn_data"]
    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "n_turns": len(rollout["turn_data"]),
        "has_logprobs": has_any_logprobs,
        "reward": rollout["reward"],
        "gt": int(sample["Y"]),
    }
