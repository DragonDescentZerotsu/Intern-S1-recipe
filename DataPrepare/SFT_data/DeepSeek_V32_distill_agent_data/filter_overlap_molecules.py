import json
import os
import re
from collections import defaultdict

def extract_smiles(text):
    # Pattern: "Drug SMILES:" followed by whitespace, then capture the SMILES until the next newline
    match = re.search(r"Drug SMILES:\s*([^\n]+)", text)
    if match:
        return match.group(1).strip()
    return None

def normalize_task_name(filename):
    # Remove extension
    name = filename.replace('.jsonl', '')
    # Remove _raw suffix if present (common in old data)
    if name.endswith('_raw'):
        name = name[:-4]
    # Remove _alpaca suffix if present
    if name.endswith('_alpaca'):
        name = name[:-7]
    return name

def get_smiles_by_task_from_new_data(directory):
    task_smiles = defaultdict(set)
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return task_smiles

    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    print(f"Scanning {len(files)} files in new data directory: {directory}")
    
    for filename in files:
        task_name = normalize_task_name(filename)
        if filename == 'metadata.json': 
            continue

        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        content = data['text']
                        s = extract_smiles(content)
                        if s:
                            task_smiles[task_name].add(s)
                except Exception as e:
                    print(f"Error reading line in {filename}: {e}")
    return task_smiles

def filter_and_save_data(input_dir, output_dir, test_task_smiles):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    print(f"Filtering {len(files)} files from {input_dir}...")
    
    total_original = 0
    total_kept = 0
    total_removed = 0

    for filename in files:
        task_name = normalize_task_name(filename)
        input_filepath = os.path.join(input_dir, filename)
        output_filepath = os.path.join(output_dir, filename)
        
        # Get the set of SMILES to exclude for this task (if any)
        exclude_smiles = test_task_smiles.get(task_name, set())
        
        file_kept = 0
        file_removed = 0
        
        with open(input_filepath, 'r') as fin, open(output_filepath, 'w') as fout:
            for line in fin:
                try:
                    data = json.loads(line)
                    # Check if SMILES is in the exclusion list
                    should_remove = False
                    
                    # Alpaca format: "instruction" field
                    if 'instruction' in data:
                        content = data['instruction']
                        s = extract_smiles(content)
                        if s and s in exclude_smiles:
                            should_remove = True
                    
                    if not should_remove:
                        fout.write(line)
                        file_kept += 1
                    else:
                        file_removed += 1
                        
                except Exception as e:
                    print(f"Error processing line in {filename}: {e}")
                    # If error, maybe keep or skip? Let's skip safely but warn
                    continue
        
        total_original += (file_kept + file_removed)
        total_kept += file_kept
        total_removed += file_removed
        print(f"Task: {task_name:<30} | Kept: {file_kept:<5} | Removed: {file_removed:<5} | Excluded Set Size: {len(exclude_smiles)}")

    print("\n" + "="*60)
    print(f"Filtering Complete.")
    print(f"Total Original Entries: {total_original}")
    print(f"Total Kept Entries:     {total_kept}")
    print(f"Total Removed Entries:  {total_removed}")
    print(f"Output Directory:       {output_dir}")
    print("="*60)

def main():
    old_data_dir = "DataPrepare/SFT_data/DeepSeek_V32_distill_agent_data/intern-s1-mini_TDC_train_Alpaca_per_task"
    new_test_dir = "DataPrepare/TDC_test_prompts_label_scaffold"
    output_dir = "DataPrepare/SFT_data/DeepSeek_V32_distill_agent_data/intern-s1-mini_TDC_train_Alpaca_per_task_scaffold_filtered"

    print("--- Loading New Test Data (SMILES to exclude) ---")
    new_test_task_smiles = get_smiles_by_task_from_new_data(new_test_dir)
    
    print("\n--- Filtering Old Data ---")
    filter_and_save_data(old_data_dir, output_dir, new_test_task_smiles)

if __name__ == "__main__":
    main()
