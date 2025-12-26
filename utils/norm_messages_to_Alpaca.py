import json
from save_jsonl import save_jsonl
from typing import Any, Dict, List
import pickle
from normalize_messages import *
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("internlm/Intern-S1-mini", trust_remote_code=True)

def get_system(tools: list = None):
    default_thinking_sys = "You are an expert reasoner with extensive experience in all areas. You approach problems through systematic thinking and rigorous reasoning. Your response should reflect deep understanding and precise logical thinking, making your solution path and reasoning clear to others. Please put your thinking process within <think>...</think> tags."
    tool_instruction = """Your response should consist of a reasoning step (**thought**) followed immediately by a function call in valid JSON format. Wrap each function call using the `<|action_start|><|plugin|>` and `<|action_end|>` tags.

**Format example:**

```
(Your thought goes here...)

<|action_start|><|plugin|>
{
    "name": "tool_name",
    "parameters": {
        "parameter1": "value1",
        "parameter2": "value2"
    }
}
<|action_end|>
```

# External Tools
You have access to these tools:"""
    if tools:
        tool_instruction += "\n" + json.dumps(tools, ensure_ascii=False, indent=2)
        return default_thinking_sys + "\n\n" + tool_instruction
    return default_thinking_sys

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def interns1_tool_blocks(tool_calls: List[Dict[str, Any]]) -> str:
    blocks = []
    for tc in tool_calls or []:
        fn = tc.get("function", {}) or {}
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")

        # arguments 可能是 JSON 字符串，也可能已经是 dict
        if isinstance(args, str):
            obj = _safe_json_loads(args)
            args_str = json.dumps(obj, ensure_ascii=False) if obj is not None else args
        else:
            args_str = json.dumps(args, ensure_ascii=False)

        blocks.append(
            "<|action_start|><|plugin|>\n"
            + json.dumps({"name": name, "parameters": json.loads(args_str)}, ensure_ascii=False)
            + "\n<|action_end|>"
        )
    return "\n".join(blocks)

def assistant_to_interns1_value(m: Dict[str, Any]) -> str:
    think = (m.get("reasoning_content") or m.get("reasoning") or "").strip()
    content = (m.get("content") or "").lstrip()
    tool_txt = interns1_tool_blocks(m.get("tool_calls") or [])

    out = ""
    if think:
        out += f"<think>\n{think}\n</think>\n"
    if content:
        out += content + "\n"
    if tool_txt:
        out += tool_txt
    return out.strip("\n")+'<|im_end|>'

def openai_messages_to_alpaca(messages: List[Dict[str, Any]], system: str = None, tools: Any = None):

    item = {"instruction": "", "input": "", "output": ""}
    item["instruction"] = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, tools=tools).rstrip('<think>')
    item["output"] = assistant_to_interns1_value(messages[-1])
    # item["output"] = assistant_to_interns1_value(messages[-1])
    return item

def split_into_steps(messages: List[Dict[str, Any]], system: str = None, tools: Any = None):
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            out.append(openai_messages_to_alpaca(messages[: i + 1], system=system, tools=tools))
    return out

if __name__ == "__main__":
    tools = [{
        'type': 'function',
        'function': {
            'name': 'describe_high_level_fg_fragments',
            'description': 'Parse SMILES string into functional groups with attachment points and mark atom ids in the SMILES string for better structure description.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'smiles': {
                        'type': 'string',
                        'description': 'The SMILES string to parse.'
                    }
                },
                'required': [
                    'smiles'
                ]
            }
        }
    }, 
    ]
    # system = get_system(tools)
    # print(system)
    messages = pickle.load(open("/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/agent/DeepSeek_V32_output/skin_reaction_1.pkl", "rb"))
    messages = normalize_DeepSeek_V32_messages(messages)
    print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools))
    sliced_messages = split_into_steps(messages, tools=tools)
    save_jsonl(sliced_messages, "/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/SFT_data/DeepSeek_V32_distill_agent_data/data_Alpaca.jsonl")