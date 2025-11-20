from huggingface_hub import HfApi
import os
from pathlib import Path
from datasets import load_dataset

current_dir = Path(__file__).parent.resolve()
cache_dir = current_dir.parent / "SFT_data" / "SFT_data" / "GPT_TDC_CLS_temp_cache"
data_path = current_dir.parent / "SFT_data" / "SFT_data" / "GPT_TDC_CLS"

cache_dir.mkdir(exist_ok=True, parents=True)
data_path.mkdir(exist_ok=True, parents=True)

ds = load_dataset("Kiria-Nozan/TDC-classification", cache_dir=str(cache_dir))

ds["train"].to_json(str(data_path / "training.jsonl"))
ds["validation"].to_json(str(data_path / "validation.jsonl"))
ds["test"].to_json(str(data_path / "test.jsonl"))
