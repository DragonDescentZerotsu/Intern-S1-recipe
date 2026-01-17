"""
把转换好的 huggingface 格式权重上传到 huggingface hub
"""

import argparse
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to Hugging Face Hub")
    parser.add_argument("--local_dir", help="Path to the local directory to upload", type=str, default='checkpoints/hub_ready/Intern-S1-mini-sft-distill-ckpt30000')  # TODO: weights to upload
    parser.add_argument("--repo_id", help="Repository ID (e.g., username/repo-name)", type=str, default='Kiria-Nozan/Intern-s1-mini-distill-dsv32-11k-samples')  # TODO: change to destination
    parser.add_argument("--create_repo", action="store_false", help="Create the repo if it doesn't exist")
    parser.add_argument("--private", action="store_true", help="Make the repo private if creating it")
    args = parser.parse_args()

    api = HfApi()
    
    if args.create_repo:
        api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
        
    print(f"Uploading {args.local_dir} to {args.repo_id}...")
    api.upload_folder(
        folder_path=args.local_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print("Upload complete!")

if __name__ == "__main__":
    main()
