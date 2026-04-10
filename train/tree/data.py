from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("DataPrepare") / "TDC_no_conflict_labels_salt_removed"
DEFAULT_SMILES_FIELD = "drug"
DEFAULT_LABEL_FIELD = "Y"


@dataclass(frozen=True)
class TdcSplitDataset:
    task: str
    split: str
    smiles: list[str]
    labels: list[int]


def normalize_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
    raise ValueError(f"Unsupported binary label value: {value!r}")


def get_split_path(task: str, split: str, data_root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    path = Path(data_root) / split / f"{task}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset split: {path}")
    return path


def list_tasks(data_root: str | Path = DEFAULT_DATA_ROOT) -> list[str]:
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    return sorted(path.stem for path in train_dir.glob("*.jsonl"))


def load_tdc_split(
    task: str,
    split: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    smiles_field: str = DEFAULT_SMILES_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> TdcSplitDataset:
    split_path = get_split_path(task, split, data_root)
    smiles: list[str] = []
    labels: list[int] = []

    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if smiles_field not in record:
                raise KeyError(f"Missing smiles field {smiles_field!r} at {split_path}:{line_number}")
            if label_field not in record:
                raise KeyError(f"Missing label field {label_field!r} at {split_path}:{line_number}")
            smiles.append(str(record[smiles_field]))
            labels.append(normalize_binary_label(record[label_field]))

    return TdcSplitDataset(
        task=task,
        split=split,
        smiles=smiles,
        labels=labels,
    )


def get_tdc_split_sample(
    task: str,
    split: str,
    sample_index: int,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    smiles_field: str = DEFAULT_SMILES_FIELD,
    label_field: str = DEFAULT_LABEL_FIELD,
) -> dict[str, Any]:
    if sample_index < 0:
        raise IndexError(f"sample_index must be non-negative, got {sample_index}")

    split_path = get_split_path(task, split, data_root)
    current_index = -1
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            current_index += 1
            if current_index != sample_index:
                continue

            record = json.loads(line)
            if smiles_field not in record:
                raise KeyError(f"Missing smiles field {smiles_field!r} at {split_path}:{line_number}")
            if label_field not in record:
                raise KeyError(f"Missing label field {label_field!r} at {split_path}:{line_number}")
            return {
                "task": task,
                "split": split,
                "sample_index": sample_index,
                "smiles": str(record[smiles_field]),
                "label": normalize_binary_label(record[label_field]),
                "record": record,
            }

    raise IndexError(
        f"sample_index {sample_index} is out of range for {split_path}. "
        f"Last available index: {current_index}"
    )
