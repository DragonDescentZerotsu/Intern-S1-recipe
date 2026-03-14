import json
import os
import glob
from pathlib import Path

def process_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            text = data.get('text', '')
            
            # Find the start of the precomputed tool results section
            split_marker = "=== Precomputed Tool Results for: "
            if split_marker in text:
                # Keep everything from the split marker onwards
                # This drops the playbook instructions placed before it
                data['text'] = text[text.find(split_marker):]
            
            f_out.write(json.dumps(data) + '\n')

def main():
    input_dir = Path("/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_prepended/Josh_origin")
    output_dir = Path("/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_prepended/Josh_playbook_removed")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train, valid, test splits
    for split in ['train', 'valid', 'test']:
        split_in_dir = input_dir / split
        split_out_dir = output_dir / split
        
        if not split_in_dir.exists():
            continue
            
        split_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all jsonl files in the split
        for input_file in split_in_dir.glob('*.jsonl'):
            output_file = split_out_dir / input_file.name
            print(f"Processing {input_file.relative_to(input_dir)} -> {output_file.relative_to(output_dir)}")
            process_file(input_file, output_file)
            
    print("Done!")

if __name__ == "__main__":
    main()
