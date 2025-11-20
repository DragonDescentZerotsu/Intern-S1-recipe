from huggingface_hub import HfApi
import os
from pathlib import Path
from datasets import load_dataset

current_dir = Path(__file__).parent.resolve()
cache_dir = current_dir.parent / "SFT_data" / "SFT_data" / "GPT_TDC_CLS_temp_cache"
data_path = current_dir.parent / "SFT_data" / "SFT_data" / "GPT_TDC_CLS_try"

ds = load_dataset("Kiria-Nozan/TDC-classification", cache_dir=str(cache_dir))



for fname in ["training.jsonl", "validation.jsonl", "test.jsonl"]:
    api.upload_file(
        path_or_fileobj=data_path/ fname,
        path_in_repo=fname,
        repo_id="Kiria-Nozan/TDC-classification",
        repo_type="dataset",
    )