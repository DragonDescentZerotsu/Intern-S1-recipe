#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def _tensor_summary(x: torch.Tensor) -> str:
    shape = tuple(x.shape)
    return f"Tensor(shape={shape}, dtype={x.dtype}, device={x.device}, numel={x.numel()})"


def _inspect(
    obj: Any,
    path: str = "root",
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 20,
) -> None:
    indent = "  " * depth

    if torch.is_tensor(obj):
        print(f"{indent}{path}: {_tensor_summary(obj)}")
        return

    if isinstance(obj, Mapping):
        print(f"{indent}{path}: dict(len={len(obj)})")
        if depth >= max_depth:
            return
        for idx, (k, v) in enumerate(obj.items()):
            if idx >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more keys)")
                break
            _inspect(v, path=f"{path}[{k!r}]", depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        seq_type = type(obj).__name__
        print(f"{indent}{path}: {seq_type}(len={len(obj)})")
        if depth >= max_depth:
            return
        for idx, item in enumerate(obj):
            if idx >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more items)")
                break
            _inspect(item, path=f"{path}[{idx}]", depth=depth + 1, max_depth=max_depth, max_items=max_items)
        return

    print(f"{indent}{path}: {type(obj).__name__} -> {obj!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the structure of a .pt file.")
    parser.add_argument("--path", default="data-partial-rollout-new-reward.pt", help="Path to .pt/.pth checkpoint file")
    parser.add_argument("--max-depth", type=int, default=4, help="Max recursive depth (default: 4)")
    parser.add_argument("--max-items", type=int, default=20, help="Max items to print per container (default: 20)")
    args = parser.parse_args()

    pt_path = Path(args.path)
    if not pt_path.exists():
        raise FileNotFoundError(f"File not found: {pt_path.resolve()}")
    if pt_path.suffix not in {".pt", ".pth"}:
        raise ValueError(f"--path should point to a .pt/.pth file, got: {pt_path}")

    print(f"Loading: {pt_path.resolve()}")
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except (pickle.UnpicklingError, RuntimeError) as e:
        raise ValueError(f"Failed to load checkpoint from {pt_path}: {e}") from e
    print(f"Top-level type: {type(data).__name__}")
    _inspect(data, max_depth=args.max_depth, max_items=args.max_items)


if __name__ == "__main__":
    main()
