# 把 train, test, valid 的 FG desc 合并成一个文件， TDC all

import json
import argparse

def merge_jsonl(input_files, output_file):
    merged = {}

    for fp in input_files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # 每行可能是 {smiles: desc} 或包含多个键值对
                merged.update(obj)

    with open(output_file, "w", encoding="utf-8") as f:
        for k, v in merged.items():
            f.write(json.dumps({k: v}, ensure_ascii=False) + "\n")

    print(f"✅ Merged {len(input_files)} files -> {output_file}")
    print(f"✅ Total unique SMILES: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in1", type=str, default="DataPrepare/shared_data/append_train.jsonl")
    parser.add_argument("--in2", type=str, default="DataPrepare/shared_data/append_test.jsonl")
    parser.add_argument("--in3", type=str, default="DataPrepare/shared_data/append_valid.jsonl")
    parser.add_argument("--in4", type=str, default="DataPrepare/shared_data/TDC_all_fg_desc_with_attach_points_and_atom_ids.jsonl")
    parser.add_argument("--out", type=str, default="DataPrepare/shared_data/TDC_all_fg_desc_with_attach_points_and_atom_ids.jsonl")
    args = parser.parse_args()

    merge_jsonl([args.in1, args.in2, args.in3, args.in4], args.out)
