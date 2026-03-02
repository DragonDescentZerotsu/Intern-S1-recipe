import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

class ModelAdapter:
    """Base adapter for model-specific formatting differences."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def stop_tokens(self) -> list[str]:
        raise NotImplementedError

    def eos_strips(self) -> list[str]:
        """Tokens to strip from end of generated text."""
        raise NotImplementedError

    def skip_special_on_decode(self) -> bool:
        """Whether to use skip_special_tokens=True when decoding."""
        return True

    def parse_tool_calls(self, text: str) -> list[dict]:
        """Parse tool calls from model output. Returns list of {name, arguments}."""
        raise NotImplementedError

    def extract_final_text(self, text: str) -> str:
        """Extract the actual response text (stripping CoT, channel markers, etc.)."""
        raise NotImplementedError

    def format_assistant_message(self, raw_text: str) -> dict:
        """Format raw model output into a message dict for apply_chat_template.
        Default: simple content field. Override for structured formats."""
        return {"role": "assistant", "content": raw_text}

    def format_tool_result_message(self, name: str, content: str, call_id: str = "call_0") -> dict:
        """Format a tool result as a message dict for apply_chat_template."""
        return {"role": "tool", "name": name, "content": content}

    def tools_as_system_suffix(self, tools: list[dict]) -> Optional[str]:
        """Optional extra text to append to system prompt."""
        return None

    def apply_chat_template_kwargs(self, tools: list[dict]) -> dict:
        """Extra kwargs passed to apply_chat_template."""
        return {"tools": tools} if tools else {}


class QwenAdapter(ModelAdapter):
    """Adapter for Qwen3 family (ChatML format)."""

    def stop_tokens(self):
        eos = self.tokenizer.eos_token or "<|im_end|>"
        return list(set([eos, "<|im_end|>", "<|endoftext|>"]))

    def eos_strips(self):
        return ["<|im_end|>", "<|endoftext|>"]

    def skip_special_on_decode(self):
        return True

    def parse_tool_calls(self, text):
        calls = []
        for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
            try:
                c = json.loads(m.group(1))
                if "name" in c: calls.append(c)
            except json.JSONDecodeError: pass
        for m in re.finditer(r'✿FUNCTION✿:\s*(\w+)\s*\n✿ARGS✿:\s*(\{.*?\})', text, re.DOTALL):
            try: calls.append({"name": m.group(1), "arguments": json.loads(m.group(2))})
            except json.JSONDecodeError: pass
        return calls

    def extract_final_text(self, text):
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return cleaned if cleaned else text

    def format_assistant_message(self, raw_text):
        return {"role": "assistant", "content": raw_text}

    def tools_as_system_suffix(self, tools):
        if not tools: return None
        lines = [
            "\n\nYou can call the following tools to help analyze molecules:\n",
        ]
        for t in tools:
            fn = t["function"]
            ps = ", ".join(f'{k}: {v.get("type","string")}'
                           for k,v in fn.get("parameters",{}).get("properties",{}).items())
            lines.append(f"- **{fn['name']}**({ps}): {fn['description']}")
        lines.append(
            '\nTo call a tool, use: <tool_call>\n{"name": "tool_name", "arguments": {"param": "value"}}\n</tool_call>'
            "\n\nWhen you have a final answer, state it as: Answer: (A) or Answer: (B)"
        )
        return "\n".join(lines)


class GptOssAdapter(ModelAdapter):
    """Adapter for OpenAI GPT-OSS (Harmony format)."""

    def stop_tokens(self):
        return ["<|call|>", "<|return|>"]

    def eos_strips(self):
        return ["<|call|>", "<|return|>", "<|end|>"]

    def skip_special_on_decode(self):
        return False

    def parse_tool_calls(self, text):
        calls = []
        for m in re.finditer(
            r'to=functions\.(\w+)\s*(?:<\|constrain\|>json)?\s*<\|message\|>\s*(\{.*)',
            text, re.DOTALL
        ):
            name = m.group(1)
            json_str = m.group(2).strip()
            for tok in ["<|call|>", "<|end|>", "<|return|>"]:
                json_str = json_str.split(tok)[0]
            try:
                args = json.loads(json_str.strip())
                calls.append({"name": name, "arguments": args})
            except json.JSONDecodeError:
                try:
                    brace_depth = 0
                    end_idx = 0
                    for i, ch in enumerate(json_str):
                        if ch == '{': brace_depth += 1
                        elif ch == '}': brace_depth -= 1
                        if brace_depth == 0 and i > 0:
                            end_idx = i + 1
                            break
                    if end_idx > 0:
                        args = json.loads(json_str[:end_idx])
                        calls.append({"name": name, "arguments": args})
                except (json.JSONDecodeError, ValueError):
                    logger.warning(f"Failed to parse GPT-OSS tool call args for {name}")
        return calls

    def extract_final_text(self, text):
        finals = re.findall(
            r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)',
            text, re.DOTALL
        )
        if finals:
            return finals[-1].strip()

        cleaned = re.sub(
            r'<\|channel\|>analysis<\|message\|>.*?(?:<\|end\|>|$)',
            '', text, flags=re.DOTALL
        )
        cleaned = re.sub(r'<\|(?:channel|message|start|end|return|constrain)\|>[^<]*', '', cleaned)
        cleaned = cleaned.strip()
        return cleaned if cleaned else text

    def _salvage_json(self, json_str):
        try:
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            pass
        try:
            brace_depth = 0
            end_idx = 0
            for i, ch in enumerate(json_str):
                if ch == '{': brace_depth += 1
                elif ch == '}': brace_depth -= 1
                if brace_depth == 0 and i > 0:
                    end_idx = i + 1
                    break
            if end_idx > 0:
                json.loads(json_str[:end_idx])
                return json_str[:end_idx]
        except (json.JSONDecodeError, ValueError):
            pass
        return "{}"

    def format_assistant_message(self, raw_text):
        msg = {"role": "assistant"}
        analysis_parts = re.findall(
            r'<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|<\|channel\|>|<\|return\|>|$)',
            raw_text, re.DOTALL
        )
        thinking = None
        if analysis_parts:
            raw_thinking = "\n".join(p.strip() for p in analysis_parts if p.strip())
            raw_thinking = re.sub(r'<\|[^|]+\|>', '', raw_thinking).strip()
            if raw_thinking:
                thinking = raw_thinking

        tool_calls = []
        for idx, m in enumerate(re.finditer(
            r'to=functions\.(\w+)\s*(?:<\|constrain\|>json)?\s*<\|message\|>\s*(\{.*?)(?=<\|call\|>|<\|end\|>|<\|channel\|>|<\|return\|>|$)',
            raw_text, re.DOTALL
        )):
            fn_name = m.group(1)
            args_str = m.group(2).strip()
            for tok in ["<|call|>", "<|end|>", "<|return|>"]:
                args_str = args_str.split(tok)[0]
            args_str = self._salvage_json(args_str.strip())
            tool_calls.append({
                "id": f"call_{idx}",
                "type": "function",
                "function": {"name": fn_name, "arguments": args_str},
            })

        # if tool_calls:
        #     msg["tool_calls"] = tool_calls

        finals = re.findall(
            r'<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>|<\|channel\|>|$)',
            raw_text, re.DOTALL
        )
        final_content = None
        if finals:
            cleaned = re.sub(r'<\|[^|]+\|>', '', finals[-1]).strip()
            if cleaned:
                final_content = cleaned

        if tool_calls:
            if thinking:
                msg["thinking"] = thinking
            msg["tool_calls"] = tool_calls
        else:
            if thinking:
                msg["thinking"] = thinking
            if final_content:
                msg["content"] = final_content
            elif not thinking:
                cleaned = re.sub(r'<\|[^|]+\|>[^<]*', '', raw_text).strip()
                msg["content"] = cleaned if cleaned else raw_text

        if "content" not in msg and "tool_calls" not in msg:
            msg["content"] = ""

        return msg

    def format_tool_result_message(self, name, content, call_id="call_0"):
        return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}

    def tools_as_system_suffix(self, tools):
        if not tools: 
            return None
        return "\n\nWhen you have a final answer, state it clearly as: Answer: (A) or Answer: (B)"

    def apply_chat_template_kwargs(self, tools):
        return {"tools": tools} if tools else {}


def get_adapter(model_name: str, tokenizer) -> ModelAdapter:
    """Factory: pick the right adapter based on model name."""
    name_lower = model_name.lower()
    if "gpt-oss" in name_lower or "gpt_oss" in name_lower:
        return GptOssAdapter(tokenizer)
    return QwenAdapter(tokenizer)


def patch_chat_template(tokenizer):
    """Fix known bugs in the GPT-OSS HF chat template for Harmony tool calling."""
    t = tokenizer.chat_template
    if not t:
        return tokenizer

    t = t.replace(
        '{{- "<|start|>assistant to=" }}\n'
        '            {{- "functions." + tool_call.name + "<|channel|>commentary " }}\n'
        '            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}',

        '{{- "<|start|>assistant<|channel|>commentary to=" }}\n'
        '            {{- "functions." + tool_call.name + " <|constrain|>" }}\n'
        '            {{- (tool_call.content_type if tool_call.content_type is defined else "json") + "<|message|>" }}',
    )
    t = t.replace(
        '{{- " to=assistant<|channel|>commentary<|message|>" + message.content|tojson + "<|end|>" }}',
        '{{- " to=assistant<|channel|>commentary<|message|>" + message.content + "<|end|>" }}',
    )
    t = t.replace(
        '{{- tool_call.arguments|tojson }}',
        '{{- tool_call.arguments }}',
    )
    tokenizer.chat_template = t
    return tokenizer
