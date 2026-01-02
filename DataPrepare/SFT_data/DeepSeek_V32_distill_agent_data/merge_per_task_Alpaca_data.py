import glob
import os
from tqdm import tqdm
from pathlib import Path

current_dir = Path(__file__).parent

def merge_jsonl_files(source_dir, output_file):
    """
    Merges all .jsonl files from the source directory into a single output file.
    """
    # Ensure output directory exists (though the user said the dir exists)
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of all jsonl files
    jsonl_files = sorted(glob.glob(os.path.join(source_dir, "*.jsonl")))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {source_dir}")
        return

    print(f"Found {len(jsonl_files)} files to merge.")
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(jsonl_files, desc="Merging files"):
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    total_lines += 1
                    
    print(f"Successfully merged {len(jsonl_files)} files into {output_file}")
    print(f"Total lines written: {total_lines}")

if __name__ == "__main__":
    source_directory = current_dir / "TDC_train_Alpaca_per_task"
    output_filepath = current_dir / "TDC_train_Alpaca_merged/TDC_train_Alpaca_merged.jsonl"
    
    merge_jsonl_files(source_directory, output_filepath)
