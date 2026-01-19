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
    return name

def get_smiles_by_task_from_old_data(directory):
    task_smiles = defaultdict(set)
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    print(f"Scanning {len(files)} files in old data directory: {directory}")
    
    for filename in files:
        task_name = normalize_task_name(filename)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'messages' in data and len(data['messages']) > 0:
                        first_msg = data['messages'][0]
                        if first_msg.get('role') == 'user':
                            content = first_msg.get('content', '')
                            s = extract_smiles(content)
                            if s:
                                task_smiles[task_name].add(s)
                except Exception as e:
                    print(f"Error reading line in {filename}: {e}")
    return task_smiles

def get_smiles_by_task_from_new_data(directory):
    task_smiles = defaultdict(set)
    files = [f for f in os.listdir(directory) if f.endswith('.jsonl')]
    print(f"Scanning {len(files)} files in new data directory: {directory}")
    
    for filename in files:
        task_name = normalize_task_name(filename)
        # Skip metadata.json or other non-task files if they happen to be picked up (though we filter by .jsonl)
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

def main():
    old_data_dir = "DataPrepare/SFT_data/DeepSeek_V32_distill_agent_data/TDC_train_raw_normalized_messages_per_task"
    new_train_dir = "DataPrepare/TDC_train_prompts_label_scaffold"
    new_test_dir = "DataPrepare/TDC_test_prompts_label_scaffold"

    print("--- Loading Old Data (per task) ---")
    old_task_smiles = get_smiles_by_task_from_old_data(old_data_dir)

    print("\n--- Loading New Train Data (per task) ---")
    new_train_task_smiles = get_smiles_by_task_from_new_data(new_train_dir)
    
    print("\n--- Loading New Test Data (per task) ---")
    new_test_task_smiles = get_smiles_by_task_from_new_data(new_test_dir)

    # Combine New Train + Test per task
    new_combined_task_smiles = defaultdict(set)
    all_tasks = set(new_train_task_smiles.keys()) | set(new_test_task_smiles.keys()) | set(old_task_smiles.keys())
    
    for task in all_tasks:
        new_combined_task_smiles[task] = new_train_task_smiles.get(task, set()) | new_test_task_smiles.get(task, set())

    print("\n" + "="*145)
    print(f"{'Task Name':<35} | {'Old':<6} | {'N.Trn':<7} | {'N.Tst':<7} | {'Ovlp.Trn':<9} | {'%Ov.Trn':<8} | {'Ovlp.Tst':<9} | {'%Ov.Tst':<8} | {'Old-Test':<9} | {'%Old-T/Trn':<10}")
    print("-" * 155)

    total_old = 0
    total_new_train = 0
    total_new_test = 0
    total_new_all = 0
    total_overlap_train = 0
    total_overlap_test = 0
    total_overlap_all = 0
    total_old_minus_test = 0

    # Sort tasks alphabetically
    sorted_tasks = sorted(list(old_task_smiles.keys()))

    for task in sorted_tasks:
        old_set = old_task_smiles[task]
        new_train_set = new_train_task_smiles.get(task, set())
        new_test_set = new_test_task_smiles.get(task, set())
        new_combined_set = new_combined_task_smiles.get(task, set())
        
        n_old = len(old_set)
        n_new_train = len(new_train_set)
        n_new_test = len(new_test_set)
        n_new_all = len(new_combined_set)
        
        overlap_train = len(old_set.intersection(new_train_set))
        overlap_test = len(old_set.intersection(new_test_set))
        overlap_all = len(old_set.intersection(new_combined_set))

        old_minus_test_set = old_set - new_test_set
        n_old_minus_test = len(old_minus_test_set)
        
        pct_train = (overlap_train / n_new_train * 100) if n_new_train > 0 else 0.0
        pct_test = (overlap_test / n_new_test * 100) if n_new_test > 0 else 0.0
        pct_all = (overlap_all / n_new_all * 100) if n_new_all > 0 else 0.0
        
        pct_old_minus_test_vs_train = (n_old_minus_test / n_new_train * 100) if n_new_train > 0 else 0.0

        print(f"{task:<35} | {n_old:<6} | {n_new_train:<7} | {n_new_test:<7} | {overlap_train:<9} | {pct_train:<7.2f}% | {overlap_test:<9} | {pct_test:<7.2f}% | {n_old_minus_test:<9} | {pct_old_minus_test_vs_train:<9.2f}%")
        
        total_old += n_old
        total_new_train += n_new_train
        total_new_test += n_new_test
        total_new_all += n_new_all
        total_overlap_train += overlap_train
        total_overlap_test += overlap_test
        total_overlap_all += overlap_all
        total_old_minus_test += n_old_minus_test

    print("-" * 155)
    print(f"{'TOTAL (Sum of Tasks)':<35} | {total_old:<6} | {total_new_train:<7} | {total_new_test:<7} | {total_overlap_train:<9} | {(total_overlap_train/total_new_train*100) if total_new_train > 0 else 0:<7.2f}% | {total_overlap_test:<9} | {(total_overlap_test/total_new_test*100) if total_new_test > 0 else 0:<7.2f}% | {total_old_minus_test:<9} | {(total_old_minus_test/total_new_train*100) if total_new_train > 0 else 0:<9.2f}%")
    print("=" * 155)
    
    # Just for completeness, Global Unique Stats (deduplicated across tasks)
    all_old_unique = set()
    for s_set in old_task_smiles.values():
        all_old_unique.update(s_set)
        
    all_new_unique = set()
    for s_set in new_combined_task_smiles.values():
        all_new_unique.update(s_set)
        
    global_overlap = len(all_old_unique.intersection(all_new_unique))
    
    print(f"\nGlobal Unique Statistics (Deduplicated across all tasks):")
    print(f"Old Unique: {len(all_old_unique)}")
    print(f"New Unique: {len(all_new_unique)}")
    print(f"Global Overlap: {global_overlap}")

if __name__ == "__main__":
    main()
