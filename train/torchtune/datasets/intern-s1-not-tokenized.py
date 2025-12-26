#
# In datasets/custom_dataset.py
#
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from pathlib import Path
from torchtune.datasets import SFTDataset, PackedDataset

current_dir = Path(__file__).parent.resolve()


def pack_tokenized_dataset(tokenizer: ModelTokenizer):
    """
    Python subset of nampdn-ai/tiny-codes. Instruct and code response pairs.
    """
    model_name = "internlm/Intern-S1-mini"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    max_length = tokenizer.model_max_length
    pad_token_id = tokenizer.pad_token_id

    TOK_PATH = current_dir.parent.parent / "SFT_data" / "SFT_data" / "TDC_SFT_data_all_tasks.arrow"

    ds = load_from_disk(str(TOK_PATH))
    ds = ds.rename_column("input_ids", "tokens")
    pack_ds = PackedDataset(ds, max_seq_len=max_length, padding_idx=pad_token_id, split_across_pack=False)

    return pack_ds