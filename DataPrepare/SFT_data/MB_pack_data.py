#!/usr/bin/env python
import argparse
from pathlib import Path

from megatron.bridge.data.datasets.utils import build_index_files
from megatron.bridge.data.datasets.packed_sequence import prepare_packed_sequence_data
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer, _HuggingFaceTokenizer

current_dir = Path(__file__).parent.resolve()

def build_packed(
    input_jsonl: Path,
    out_dir: Path,
    tokenizer_model: str,
    packed_size: int = 1024,
    seq_length: int = 1024,
    workers: int = 2,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 先把 jsonl 的 memmap 索引建好，用你想要的 worker 数
    # 这个就是你日志里那句 "Processing ... using 2 workers" 的真正来源，我们这里自己调一次，
    # 并且把 workers 设置成你想要的数，这样后面再读这个 jsonl 的时候就不会再建索引了。
    # https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.data.datasets.utils.html#bridge.data.datasets.utils.build_index_files
    build_index_files(
        dataset_paths=[str(input_jsonl)],
        newline_int=10,
        workers=workers,
        index_mapping_dir=str(out_dir),  # 索引也放到你指定的目录里
    )

    # 2) 准备 tokenizer，跟训练时配置保持一致
    # https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.training.tokenizers.tokenizer.html#bridge.training.tokenizers.tokenizer.build_tokenizer
    tok_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )
    # tokenizer = build_tokenizer(tok_cfg)
    tokenizer = _HuggingFaceTokenizer(tokenizer_model)

    # 3) 真正做 packed
    # https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.data.datasets.packed_sequence.html#bridge.data.datasets.packed_sequence.prepare_packed_sequence_data
    output_path = out_dir / f"{input_jsonl.stem}_{packed_size}.npy"
    metadata_path = out_dir / f"packed_meta_{packed_size}.jsonl"

    # 你训练里写的那些 dataset_kwargs 也可以在这传进来，保证两边规则一致
    dataset_kwargs = dict(
        answer_only_loss=True,
        chat=False,
        use_hf_tokenizer_chat_template=False,
        label_key="output",
        truncation_field="input",
        add_eos=False,
    )

    prepare_packed_sequence_data(
        input_path=input_jsonl,
        output_path=output_path,
        output_metadata_path=metadata_path,
        packed_sequence_size=packed_size,
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        seed=0,
        packing_algorithm="first_fit_shuffle",
        dataset_kwargs=dataset_kwargs,
    )

    print(f"[OK] packed data: {output_path}")
    print(f"[OK] packed metadata: {metadata_path}")
    return output_path, metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed-size", type=int, default=1024)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=120,
                        help="how many processes to build memmap index with")
    args = parser.parse_args()

    build_packed(
        input_jsonl=current_dir / 'SFT_data' / 'GPT_TDC_CLS' / 'training.jsonl',
        out_dir=current_dir / 'SFT_data'/ 'GPT_TDC_CLS' / 'packed' / 'internlm' / 'Intern-S1-mini',
        tokenizer_model="internlm/Intern-S1-mini",
        packed_size=args.packed_size,
        seq_length=args.seq_length,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
