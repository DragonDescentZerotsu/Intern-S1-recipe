#!/usr/bin/env python3
"""Patch vLLM GPT-OSS Harmony parsing for legacy SFT tool-call completions.

The affected SFT model often emits legacy/incomplete Harmony fragments such as:

  <|channel|>analysis...<|end|>
  <|start|>assistant to=functions.foo<|channel|>commentary json<|message|>{...}<|call|><|call|>...

vLLM's GPT-OSS Chat path parses generated assistant tokens as standalone Harmony
messages and raises before the OpenAI tool parser can recover. This patch keeps
the strict parser as the first path and adds a narrow fallback that:

  * prefixes a missing assistant start header when generation begins at channel;
  * rewrites the old tool header order to current Harmony format;
  * truncates duplicate garbage after the first <|call|> assistant action.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import sysconfig
from pathlib import Path


RELATIVE_TARGET = Path("entrypoints/openai/parser/harmony_utils.py")
BACKUP_SUFFIX = ".bak-gptoss-sft"

OLD_FUNC = '''def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser
'''

NEW_FUNC = '''def _parse_output_into_messages_strict(token_ids: Iterable[int]) -> StreamableParser:
    parser = get_streamable_parser_for_assistant()
    for token_id in token_ids:
        parser.process(token_id)
    return parser


def _normalize_legacy_gptoss_sft_tokens(token_ids: Sequence[int]) -> list[int]:
    """Best-effort compatibility for legacy GPT-OSS SFT Harmony fragments."""
    text = get_encoding().decode(list(token_ids))

    # Some fine-tuned checkpoints continue the prompt's trailing
    # "<|start|>assistant" and begin directly with a channel tag.
    if text.startswith("<|channel|>"):
        text = "<|start|>assistant" + text
    elif not text.startswith("<|start|>"):
        text = "<|start|>assistant<|channel|>final<|message|>" + text

    # Older GPT-OSS templates rendered the recipient before the channel:
    #   <|start|>assistant to=functions.foo<|channel|>commentary json<|message|>
    # Current Harmony expects:
    #   <|start|>assistant<|channel|>commentary to=functions.foo <|constrain|>json<|message|>
    text = re.sub(
        r"<\\|start\\|>assistant\\s+to=(functions\\.[^<\\s]+)"
        r"<\\|channel\\|>commentary\\s+(?:<\\|constrain\\|>)?json<\\|message\\|>",
        r"<|start|>assistant<|channel|>commentary to=\\1 <|constrain|>json<|message|>",
        text,
    )

    # A tool-call assistant action should stop at the first <|call|>. The
    # legacy SFT model often continues with repeated <|call|> and JSON chunks,
    # which makes the Harmony parser expect a new <|start|> and then fail.
    first_call = text.find("<|call|>")
    if first_call != -1:
        text = text[: first_call + len("<|call|>")]

    return get_encoding().encode(text, allowed_special="all")


def parse_output_into_messages(token_ids: Iterable[int]) -> StreamableParser:
    token_ids_list = list(token_ids)
    try:
        return _parse_output_into_messages_strict(token_ids_list)
    except Exception as strict_exc:
        try:
            normalized = _normalize_legacy_gptoss_sft_tokens(token_ids_list)
            return _parse_output_into_messages_strict(normalized)
        except Exception:
            raise strict_exc
'''


def resolve_target(explicit_target: str | None = None) -> Path:
    if explicit_target:
        return Path(explicit_target).expanduser().resolve()

    spec = importlib.util.find_spec("vllm")
    if spec and spec.submodule_search_locations:
        target = Path(next(iter(spec.submodule_search_locations))) / RELATIVE_TARGET
        if target.exists():
            return target.resolve()

    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        target = Path(purelib) / "vllm" / RELATIVE_TARGET
        if target.exists():
            return target.resolve()

    raise FileNotFoundError(
        "Could not locate vLLM harmony_utils.py in the current Python environment. "
        "Run this script with the target conda env's python, or pass --target."
    )


def ensure_required_imports(text: str) -> str:
    if "import re\n" not in text:
        text = text.replace("import datetime\n", "import datetime\nimport re\n", 1)

    if "from collections.abc import Iterable, Sequence\n" not in text:
        text = text.replace(
            "from collections.abc import Iterable\n",
            "from collections.abc import Iterable, Sequence\n",
            1,
        )

    return text


def patch_target(target: Path, dry_run: bool = False) -> None:
    text = target.read_text()
    if "_normalize_legacy_gptoss_sft_tokens" in text:
        print(f"Already patched: {target}")
        return

    text = ensure_required_imports(text)

    if OLD_FUNC not in text:
        raise RuntimeError("Could not find expected parse_output_into_messages block.")

    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if dry_run:
        print(f"Would patch: {target}")
        print(f"Would create backup if missing: {backup}")
        return

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup written: {backup}")

    target.write_text(text.replace(OLD_FUNC, NEW_FUNC, 1))
    print(f"Patched: {target}")


def restore_target(target: Path, dry_run: bool = False) -> None:
    backup = target.with_suffix(target.suffix + BACKUP_SUFFIX)
    if not backup.exists():
        raise FileNotFoundError(f"Backup not found: {backup}")

    if dry_run:
        print(f"Would restore: {target}")
        print(f"From backup: {backup}")
        return

    shutil.copy2(backup, target)
    print(f"Restored: {target}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch the vLLM GPT-OSS Harmony parser in the current Python "
            "environment for legacy SFT tool-call completions."
        )
    )
    parser.add_argument(
        "--target",
        help=(
            "Explicit path to vllm/entrypoints/openai/parser/harmony_utils.py. "
            "By default the script locates vLLM in the current Python environment."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the target and backup paths without modifying files.",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help=f"Restore from the {BACKUP_SUFFIX} backup instead of applying the patch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = resolve_target(args.target)
    if not target.exists():
        raise FileNotFoundError(f"Target file not found: {target}")

    print(f"Python executable: {sys.executable}")
    print(f"Target file: {target}")

    if args.restore:
        restore_target(target, args.dry_run)
    else:
        patch_target(target, args.dry_run)


if __name__ == "__main__":
    main()
