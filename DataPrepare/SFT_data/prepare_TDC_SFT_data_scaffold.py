
import json
import os
from pathlib import Path
from tqdm import tqdm

def main():
    # Define directories
    current_dir = Path(__file__).parent.resolve() # DataPrepare/SFT_data
    project_root = current_dir.parent.parent # /data1/tianang/Projects/Intern-S1/
    
    input_dir = project_root / 'DataPrepare' / 'TDC_train_prompts_label_scaffold'
    # Output to SFT_data/SFT_data/TDC_SFT_data_scaffold_all.jsonl
    output_dir = current_dir / 'SFT_data'
    output_file = output_dir / 'TDC_SFT_data_scaffold_all.jsonl'
    
    # Ensure output directory exists (though SFT_data should exist)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input Directory: {input_dir}")
    print(f"Output File: {output_file}")
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # Get list of all jsonl files
    jsonl_files = list(input_dir.glob('*.jsonl'))
    
    print(f"Found {len(jsonl_files)} JSONL files to process.")
    
    total_samples = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in tqdm(jsonl_files, desc="Processing files"):
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        
                        # Replace the long prompt suffix with 'Answer:'
                        cot_suffix = 'Please think step by step and then put ONLY your final choice ((A) or (B)) after "Answer:"'
                        text = text.replace(cot_suffix, 'Answer:')
                        
                        y_label = data.get('Y')
                        
                        if y_label == 1:
                            output_val = "(B)"
                        elif y_label == 0:
                            output_val = "(A)"
                        else:
                            print(f"Warning: Unexpected label {y_label} in file {input_file.name}. Skipping line.")
                            continue
                            
                        alpaca_entry = {
                            "instruction": text,
                            "input": "",
                            "output": output_val
                        }
                        
                        f_out.write(json.dumps(alpaca_entry, ensure_ascii=False) + '\n')
                        total_samples += 1
                        
                    except json.JSONDecodeError:
                        print(f"Warning: Failed to decode JSON in file {input_file.name}. Skipping line.")
                    except Exception as e:
                         print(f"Error processing line in {input_file.name}: {e}")

    print(f"Processing complete. Total {total_samples} samples saved to {output_file}")

if __name__ == "__main__":
    main()
