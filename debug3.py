import json

# Path to the JSONL file
file_path = "/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/DataPrepare/SFT_data/SFT_data/GPT/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/TDC_SFT_data_binary_Scaffold_wo_herg-c_ToxCast_butkiewicz/training.jsonl"

count_A = 0
count_B = 0
total_lines = 0

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        total_lines += 1
        try:
            data = json.loads(line.strip())
            output = data.get('output', '')
            if '(A)' in output:
                count_A += 1
            if '(B)' in output:
                count_B += 1
        except json.JSONDecodeError as e:
            print(f"Error parsing line {total_lines}: {e}")

print(f"Total lines: {total_lines}")
print(f"Lines with (A) in output: {count_A}")
print(f"Lines with (B) in output: {count_B}")
