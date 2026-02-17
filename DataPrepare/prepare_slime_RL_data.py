"""
Prepare Slime RL data from TDC training prompts.

Reads JSONL files from TDC_train_prompts_label_sm_wo_herg-c_ToxCast_butkiewicz/,
adds a "metadata" field with SMILES, label, task_name, in_task_id,
and saves per-task JSONL files to Slime_RL_data/by_task/.
"""

import json
import os
import re

split = 'test'

SRC_DIR = os.path.join(os.path.dirname(__file__),
                       f"TDC_{split}_prompts_label_sm_wo_herg-c_ToxCast_butkiewicz")
DST_DIR = os.path.join(os.path.dirname(__file__),
                       "Slime_RL_data", "by_task", split)


def extract_smiles(text: str) -> str:
    """Extract SMILES from 'Drug SMILES: <SMILES>\n' pattern in text."""
    match = re.search(r"Drug SMILES: (.+?)\\n", text)
    if match:
        return match.group(1).strip()
    # fallback: try with real newline
    match = re.search(r"Drug SMILES: (.+?)\n", text)
    if match:
        return match.group(1).strip()
    return ""


def process_file(src_path: str, task_name: str, dst_path: str):
    """Process a single task JSONL file."""
    records = []
    with open(src_path, "r", encoding="utf-8") as f:
        for in_task_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["text"]
            y = item["Y"]

            smiles = extract_smiles(text)
            label = "(B)" if y == 1 else "(A)"

            new_item = {
                "text": text,
                "Y": y,
                "metadata": {
                    "SMILES": smiles,
                    "label": label,
                    "task_name": task_name,
                    "in_task_id": in_task_id,
                },
            }
            records.append(new_item)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main():
    total = 0
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".jsonl"))
    for fname in files:
        task_name = fname.replace(".jsonl", "")
        src_path = os.path.join(SRC_DIR, fname)
        dst_path = os.path.join(DST_DIR, fname)
        n = process_file(src_path, task_name, dst_path)
        print(f"[{task_name}] {n} records written to {dst_path}")
        total += n
    print(f"\nDone. Total {total} records across {len(files)} tasks.")


if __name__ == "__main__":
    main()
