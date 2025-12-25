import json
from typing import Any, Dict, List

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
    return out.strip("\n")

def openai_messages_to_sharegpt(messages: List[Dict[str, Any]], system: str = None, tools: Any = None):
    conv = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            conv.append({"from": "human", "value": m.get("content", "")})
        elif role == "assistant":
            conv.append({"from": "gpt", "value": assistant_to_interns1_value(m)})
        elif role == "tool":
            conv.append({"from": "observation", "value": m.get("content", "")})
        else:
            raise ValueError(f"unsupported role: {role}")

    item = {"conversations": conv}
    if system is not None:
        item["system"] = system
    if tools is not None:
        # LF 文档里 tools 是一个字符串（JSON 串） :contentReference[oaicite:3]{index=3}
        item["tools"] = json.dumps(tools, ensure_ascii=False)
    return item

def split_into_steps(messages: List[Dict[str, Any]], system: str = None, tools: Any = None):
    out = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            out.append(openai_messages_to_sharegpt(messages[: i + 1], system=system, tools=tools))
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
    system = get_system(tools)
    print(system)