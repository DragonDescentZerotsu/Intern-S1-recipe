from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_PROMPT_ROOT = Path("DataPrepare") / "TDC_valid_prompts_label_scaffold"

OPTION_PATTERN = re.compile(
    r"\(A\)\s*(.*?)\s*\(B\)\s*(.*?)(?:\nDrug SMILES:|\nPlease think|$)",
    re.DOTALL,
)


def load_task_label_semantics(
    task: str,
    prompt_root: str | Path = DEFAULT_PROMPT_ROOT,
) -> dict[int, dict[str, str]] | None:
    prompt_path = Path(prompt_root) / f"{task}.jsonl"
    if not prompt_path.exists():
        return None

    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_text = str(record.get("text", ""))
            match = OPTION_PATTERN.search(prompt_text)
            if match is None:
                return None
            option_a_text = " ".join(match.group(1).split())
            option_b_text = " ".join(match.group(2).split())
            return {
                0: {
                    "option": "A",
                    "text": option_a_text,
                },
                1: {
                    "option": "B",
                    "text": option_b_text,
                },
            }
    return None
