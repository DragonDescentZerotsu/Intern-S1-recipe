# prepare_dataset.py
import os
from datasets import load_dataset
from transformers import AutoTokenizer
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2

MODEL_NAME = "internlm/Intern-S1-mini-FP8"
INPUT_JSON = "SFT_data/TDC_SFT_data_all_tasks.json"   # 顶层数组JSON
OUT_DIR    = "SFT_data/TDC_SFT_data_all_tasks.arrow"
<<<<<<< HEAD
MAX_LEN    = 2048
=======
=======
from pathlib import Path

current_dir = Path(__file__).parent.resolve()

MODEL_NAME = "internlm/Intern-S1-mini-FP8"
INPUT_JSON = current_dir/"SFT_data"/"TDC_SFT_data_all_tasks.json"   # 顶层数组JSON
OUT_DIR    = current_dir/"SFT_data"/"TDC_SFT_data_all_tasks.arrow"
>>>>>>> 14496c5f75f86b8064c818c7b9b05f570038301f
MAX_LEN    = None
enable_thinking = False

# try_dict = {'instruction': 'Instructions: Answer the following question about drug properties.\nContext: Repetitive exposure to a chemical agent can induce an immune reaction in inherently susceptible individuals that leads to skin sensitization.\nQuestion: Given a drug SMILES string, predict whether it\n(A) does not cause a skin reaction (B) causes a skin reaction\nDrug SMILES: <SMILES>CCCOc1ccc(Br)c(C(=O)c2ccc(OC)cc2O)c1</SMILES>\nAnswer:',
#  'input': '',
#  'output': '(A)'}
>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2

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
<<<<<<< HEAD
        messages_prompt, tokenize=True, add_generation_prompt=False
=======
        messages_prompt, tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking
    )

    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2
    )

    # 3) full（sys+user+assistant）
    messages_full = messages_prompt + [{"role": "assistant", "content": record["output"]}]
<<<<<<< HEAD
    full_ids = tokenizer.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False
    )

=======
    # full_ids = tokenizer.apply_chat_template(
    #     messages_full, tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking
    # )

    full_text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking
    )

    delta_text = full_text[len(prompt_text):]
    delta_ids = tokenizer.encode(delta_text, add_special_tokens=False)
    full_ids = prompt_ids + delta_ids
    # print(full_ids==tokenizer.apply_chat_template(
    #     messages_full, tokenize=True, add_generation_prompt=False, enable_thinking=enable_thinking
    # ))

>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2
    # 4) 只监督assistant段；可选EOS
    if add_eos and tokenizer.eos_token_id is not None and (len(full_ids) == 0 or full_ids[-1] != tokenizer.eos_token_id):
        full_ids = full_ids + [tokenizer.eos_token_id]

    input_ids = full_ids
<<<<<<< HEAD
    labels    = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
=======
    labels    = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]  # delta_ids
>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2

    if max_len is not None and len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels    = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

<<<<<<< HEAD
    raw = load_dataset("json", data_files=INPUT_JSON, split="train")  # 顶层数组JSON可直接读
    ds_tok = raw.map(
        lambda ex: build_ids_and_labels(ex, tokenizer, add_eos=True, max_len=MAX_LEN),
        remove_columns=raw.column_names,
        num_proc=8,
        desc="Tokenizing with chat template",
    )
    ds_tok.save_to_disk(OUT_DIR)  # Arrow落盘
=======
    # try_out = build_ids_and_labels(try_dict, tokenizer, add_eos=True, max_len=MAX_LEN)

<<<<<<< HEAD
    raw = load_dataset("json", data_files=INPUT_JSON, split="train")  # 顶层数组JSON可直接读
=======
    raw = load_dataset("json", data_files=str(INPUT_JSON), split="train")  # 顶层数组JSON可直接读
>>>>>>> 14496c5f75f86b8064c818c7b9b05f570038301f
    ds_tok = raw.map(
        lambda ex: build_ids_and_labels(ex, tokenizer, add_eos=False, max_len=MAX_LEN),
        remove_columns=raw.column_names,
        num_proc=63,
        desc="Tokenizing with chat template",
    )
    print("Saveing tokenized dataset")
<<<<<<< HEAD
    ds_tok.save_to_disk(OUT_DIR)  # Arrow落盘
=======
    ds_tok.save_to_disk(str(OUT_DIR))  # Arrow落盘
>>>>>>> 14496c5f75f86b8064c818c7b9b05f570038301f
>>>>>>> 90bfe00c0d6049e0720b1d1d18e15442c13465f2
    print("Saved tokenized dataset to:", OUT_DIR)

if __name__ == "__main__":
    main()
