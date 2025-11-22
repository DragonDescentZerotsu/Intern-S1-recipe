# prepare_dataset.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "internlm/Intern-S1-mini-FP8"
INPUT_JSON = "SFT_data/TDC_SFT_data_all_tasks.json"   # 顶层数组JSON
OUT_DIR    = "SFT_data/TDC_SFT_data_all_tasks.arrow"
MAX_LEN    = 2048

def build_ids_and_labels(record, tokenizer, add_eos=True, max_len=None):
    # 1) 组织messages
    user_text = (record["instruction"] + ("\n" + record["input"] if record.get("input") else "")).strip()
    messages_prompt = []
    sys_text = (record.get("system") or "").strip()
    if sys_text:
        messages_prompt.append({"role": "system", "content": sys_text})
    messages_prompt.append({"role": "user", "content": user_text})

    # 2) prompt（仅sys+user）
    prompt_ids = tokenizer.apply_chat_template(
        messages_prompt, tokenize=True, add_generation_prompt=False
    )

    # 3) full（sys+user+assistant）
    messages_full = messages_prompt + [{"role": "assistant", "content": record["output"]}]
    full_ids = tokenizer.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False
    )

    # 4) 只监督assistant段；可选EOS
    if add_eos and tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = full_ids + [tokenizer.eos_token_id]

    input_ids = full_ids
    labels    = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels    = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset("json", data_files=INPUT_JSON, split="train")  # 顶层数组JSON可直接读
    ds_tok = raw.map(
        lambda ex: build_ids_and_labels(ex, tokenizer, add_eos=True, max_len=MAX_LEN),
        remove_columns=raw.column_names,
        num_proc=8,
        desc="Tokenizing with chat template",
    )
    ds_tok.save_to_disk(OUT_DIR)  # Arrow落盘
    print("Saved tokenized dataset to:", OUT_DIR)

if __name__ == "__main__":
    main()
