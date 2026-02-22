import os
import random

src_dir = '/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/DataPrepare/Slime_RL_data/by_task/test'
dst_dir = '/vast/projects/xia6/apex-gen/tianang/projects/Intern-S1/DataPrepare/Slime_RL_data/by_task/test_debug'

os.makedirs(dst_dir, exist_ok=True)

for filename in os.listdir(src_dir):
    if filename.endswith(".jsonl"):
        src_path = os.path.join(src_dir, filename)
        base_name = filename[:-6] # remove .jsonl
        dst_filename = f"{base_name}_debug.jsonl"
        dst_path = os.path.join(dst_dir, dst_filename)
        
        with open(src_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # random sample 8 lines, if less than 8 take all
        sampled_lines = random.sample(lines, min(8, len(lines)))
        
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.writelines(sampled_lines)

print("Debug datasets created successfully.")
