import os

target_dir_name = "TDC_train_Alpaca" # "TDC_train_raw_normalized_messages"
script_dir = os.path.dirname(os.path.abspath(__file__))
full_target_dir = os.path.join(script_dir, target_dir_name)

total_lines = 0

if not os.path.exists(full_target_dir):
    print(f"Directory not found: {full_target_dir}")
    exit(1)

print(f"Counting lines in files under: {full_target_dir}")

files = sorted(os.listdir(full_target_dir))
for filename in files:
    filepath = os.path.join(full_target_dir, filename)
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
                print(f"{filename}: {lines}")
                total_lines += lines
        except Exception as e:
            print(f"Error reading {filename}: {e}")

print(f"Total lines: {total_lines}")
