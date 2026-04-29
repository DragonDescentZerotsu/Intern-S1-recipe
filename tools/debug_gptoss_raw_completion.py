#!/usr/bin/env python3
"""Inspect raw GPT-OSS completions without vLLM Chat/Harmony parsing."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

from openai import OpenAI
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    module_path = PROJECT_ROOT / "test" / "test_tdc_via_api_F1_TRIM.py"
    spec = importlib.util.spec_from_file_location("tdc_trim_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_prompt(args: argparse.Namespace) -> tuple[str, Any]:
    tdc = load_eval_module()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )

    messages = [{"role": "user", "content": tdc.render_user_text(args.task, args.smiles)}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if args.enable_thinking is not None:
        kwargs["enable_thinking"] = args.enable_thinking
    if args.tool_mode != "none":
        kwargs["tools"] = tdc.get_trim_tools_for_mode(args.tool_mode, api_style="chat")

    return tokenizer.apply_chat_template(messages, **kwargs), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:9002/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default="", help="Tokenizer path/name; defaults to --model.")
    parser.add_argument("--task", default="DILI")
    parser.add_argument(
        "--smiles",
        default="CC(CO)NC(=O)C1C=C2c3cccc4[nH]cc(c34)CC2N(C)C1",
    )
    parser.add_argument("--tool-mode", choices=["none", "properties", "similar", "both"], default="similar")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    prompt, tokenizer = render_prompt(args)
    print("=== PROMPT TAIL ===")
    print(repr(prompt[-1200:]))

    client = OpenAI(api_key=args.api_key, base_url=args.api_base, max_retries=0, timeout=600.0)
    response = client.completions.create(
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        extra_body={
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            "return_tokens_as_token_ids": True,
        },
    )

    choice = response.choices[0]
    text = choice.text or ""
    print("\n=== RAW COMPLETION TEXT ===")
    print(repr(text))

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print("\n=== RE-ENCODED FIRST TOKENS ===")
    for token_id in token_ids[:40]:
        print(token_id, repr(tokenizer.decode([token_id], skip_special_tokens=False)))


if __name__ == "__main__":
    main()
