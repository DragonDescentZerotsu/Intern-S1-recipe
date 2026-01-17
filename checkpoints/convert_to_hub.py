"""
把本地的 llamafactory 获得的训练得到的权重转换成能够上传到 huggingface 的格式
"""

import os
import shutil
import argparse
import fnmatch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert LlamaFactory checkpoint to Hugging Face Hub ready format.")
    parser.add_argument("--src", type=str, required=True, help="Path to the source checkpoint directory.")
    parser.add_argument("--dst", type=str, required=True, help="Path to the destination directory.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    src_dir = args.src
    dst_dir = args.dst
    
    # Define patterns to exclude
    ignore_patterns = [
        "global_step*",
        "latest",
        "rng_state*",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin"
    ]
    
    if not os.path.exists(src_dir):
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return

    if os.path.exists(dst_dir):
        print(f"Warning: Destination directory '{dst_dir}' already exists.")
        # We can continue, effectively merging or just failing if files collide. 
        # For safety let's just proceed, assuming user knows what they are doing or it's empty.
    else:
        os.makedirs(dst_dir)
        print(f"Created destination directory '{dst_dir}'.")

    files_to_copy = []
    skipped_items = []

    for item in os.listdir(src_dir):
        # Check if item matches any ignore pattern
        should_ignore = False
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(item, pattern):
                should_ignore = True
                break
        
        if should_ignore:
            skipped_items.append(item)
            continue
        
        # We process files and potentially directories if they aren't ignored
        # But typically only files are needed for the Hub model weights (except maybe feature extractor dirs? usually flat though)
        # Based on inspection, we only expect files.
        files_to_copy.append(item)

    print(f"Found {len(files_to_copy)} items to copy.")
    print(f"Skipping {len(skipped_items)} items: {skipped_items}")

    for item in tqdm(files_to_copy, desc="Copying files"):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isdir(src_path):
            # If it's a directory and not ignored, verify if we should copy it recursively
            # For now, let's copy recursively using shutil.copytree if it doesn't exist
            if not os.path.exists(dst_path):
                shutil.copytree(src_path, dst_path)
            else:
                 print(f"Skipping directory {item} as it already exists in destination.")
        else:
            shutil.copy2(src_path, dst_path)

    print(f"Conversion complete. Files ready in '{dst_dir}'.")

if __name__ == "__main__":
    main()
