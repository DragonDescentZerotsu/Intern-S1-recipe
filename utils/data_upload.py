from huggingface_hub import HfApi
import os
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
data_path = current_dir.parent / "SFT_data" / "SFT_data" / "GPT_TDC_CLS"

api = HfApi(token=os.getenv("HF_TOKEN"))

for fname in ["training.jsonl", "validation.jsonl", "test.jsonl"]:
    api.upload_file(
        path_or_fileobj=data_path/ fname,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
    )