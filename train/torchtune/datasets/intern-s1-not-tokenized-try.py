#
# In datasets/custom_dataset.py
#
from torchtune.datasets import SFTDataset, PackedDataset, alpaca_dataset
from torchtune.data import InputOutputToMessages
from torchtune.modules.tokenizers import ModelTokenizer
from transformers import AutoProcessor, AutoTokenizer

# Example builder function for a custom code instruct dataset not in torchtune, but using
# different dataset building blocks from torchtune

model_name = "internlm/Intern-S1-mini"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

alpacadataset = alpaca_dataset()

def tiny_codes(tokenizer: ModelTokenizer, packed: bool = True):
    """
    Python subset of nampdn-ai/tiny-codes. Instruct and code response pairs.
    """
    ds = SFTDataset(
        model_transform=tokenizer,
        source="nampdn-ai/tiny-codes",
        message_transform=InputOutputToMessages(
            column_map={"input": "prompt", "output": "response"},
        ),
        filter_fn=lambda x: x["language"] == "python",
        split="train",
    )
    if packed:
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=False)
    else:
        return ds