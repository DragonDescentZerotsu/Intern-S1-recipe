#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, load_tdc_split
from rf_pipeline import load_model_bundle, select_reasoning_trees_with_model_bundle
from select_reasoning_trees import build_reasoning_schema, resolve_bundle_path, resolve_task, resolve_output_path

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate reasoning-tree traces for every sample in one or more TDC splits."
    )
    parser.add_argument("--bundle", default=None, help="Path to model_bundle.pkl")
    parser.add_argument("--bundle-root", default=str(Path(__file__).resolve().parent / "bundles"), help="Root directory for exported bundles")
    parser.add_argument("--experiment-name", default=None, help="Experiment name under bundle-root")
    parser.add_argument("--task", default=None, help="Task name under bundle-root/<experiment>")
    parser.add_argument("--feature-set", default=None, help="Feature-set directory name. If omitted, infer when unique.")
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated splits to process. Use 'all' to process train and valid.",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory of TDC JSONL splits")
    parser.add_argument("--smiles-key", default=DEFAULT_SMILES_FIELD, help="SMILES field name in the JSONL files")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_FIELD, help="Label field name in the JSONL files")
    parser.add_argument("--class-index", type=int, default=None, help="Optional class index to explain")
    parser.add_argument("--shap-top-k", type=int, default=30, help="Number of SHAP-important features to consider")
    parser.add_argument("--max-trees", type=int, default=5, help="Maximum number of trees to keep per sample")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of samples to process in each batch",
    )
    parser.add_argument(
        "--require-forest-correct",
        dest="require_forest_correct",
        action="store_true",
        help="Keep samples only when the full forest prediction matches the true label.",
    )
    parser.add_argument(
        "--allow-forest-incorrect",
        dest="require_forest_correct",
        action="store_false",
        help="Allow selecting trees even when the full forest prediction is wrong.",
    )
    parser.add_argument(
        "--output-root",
        default=str(Path(__file__).resolve().parent / "tree_reasoning_processes"),
        help="Root directory for saved reasoning-process payloads.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose output JSON already exists for the requested split/sample/top-k combination.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress display.",
    )
    parser.set_defaults(require_forest_correct=True)
    return parser.parse_args()


def parse_splits(value: str) -> list[str]:
    split_names = [item.strip() for item in value.split(",") if item.strip()]
    if not split_names:
        raise ValueError("At least one split must be provided")
    if "all" in split_names:
        return ["train", "valid"]
    return split_names


def batch_ranges(total_size: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [
        (start_index, min(start_index + batch_size, total_size))
        for start_index in range(0, total_size, batch_size)
    ]


def sample_output_prefix(split: str, sample_index: int, shap_top_k: int) -> str:
    return f"{split}_sample_{sample_index}__top{shap_top_k}__trees"


def output_dir_for_task(
    *,
    output_root: Path,
    experiment_name: str,
    task: str,
    feature_set_name: str,
) -> Path:
    return output_root / experiment_name / task / feature_set_name


def has_existing_output(
    *,
    output_dir: Path,
    split: str,
    sample_index: int,
    shap_top_k: int,
) -> bool:
    prefix = sample_output_prefix(split, sample_index, shap_top_k)
    return any(output_dir.glob(f"{prefix}*.json"))


class BatchProgress:
    def __init__(self, *, enabled: bool, split: str, total: int) -> None:
        self.enabled = enabled
        self.split = split
        self.total = total
        self.current = 0
        self.progress_bar = None
        if not enabled:
            return
        if tqdm is not None:
            self.progress_bar = tqdm(
                total=total,
                desc=f"{split} samples",
                unit="sample",
                leave=True,
            )
        else:
            print(f"[{split}] starting {total} samples", file=sys.stderr)

    def update(self, count: int) -> None:
        if count <= 0:
            return
        self.current += count
        if not self.enabled:
            return
        if self.progress_bar is not None:
            self.progress_bar.update(count)
            return
        print(f"[{self.split}] {self.current}/{self.total}", file=sys.stderr)

    def close(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()


def main() -> int:
    args = parse_args()
    bundle_path = resolve_bundle_path(args)
    task = resolve_task(args, bundle_path)
    model_bundle = load_model_bundle(bundle_path)
    feature_set_name = str(model_bundle.get("feature_set_name") or "unknown_feature_set")
    experiment_name = str(args.experiment_name or bundle_path.resolve().parents[2].name)

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    else:
        output_root = output_root.resolve()

    output_dir = output_dir_for_task(
        output_root=output_root,
        experiment_name=experiment_name,
        task=task,
        feature_set_name=feature_set_name,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    save_args = argparse.Namespace(
        output_json=None,
        experiment_name=experiment_name,
        output_root=str(output_root),
        task=task,
    )

    summary_rows: list[dict[str, object]] = []
    total_saved = 0
    total_skipped_existing = 0

    for split in parse_splits(args.splits):
        dataset = load_tdc_split(
            task=task,
            split=split,
            data_root=args.data_root,
            smiles_field=args.smiles_key,
            label_field=args.label_key,
        )

        split_saved = 0
        split_skipped_existing = 0
        split_selected_tree_total = 0
        progress = BatchProgress(
            enabled=not args.no_progress,
            split=split,
            total=len(dataset.smiles),
        )

        try:
            for start_index, end_index in batch_ranges(len(dataset.smiles), args.batch_size):
                batch_indices = list(range(start_index, end_index))
                if args.skip_existing:
                    pending_indices = [
                        sample_index
                        for sample_index in batch_indices
                        if not has_existing_output(
                            output_dir=output_dir,
                            split=split,
                            sample_index=sample_index,
                            shap_top_k=args.shap_top_k,
                        )
                    ]
                    skipped_in_batch = len(batch_indices) - len(pending_indices)
                    split_skipped_existing += skipped_in_batch
                    progress.update(skipped_in_batch)
                else:
                    pending_indices = batch_indices

                if not pending_indices:
                    continue

                batch_smiles = [dataset.smiles[sample_index] for sample_index in pending_indices]
                batch_labels = [dataset.labels[sample_index] for sample_index in pending_indices]
                batch_results = select_reasoning_trees_with_model_bundle(
                    smiles_list=batch_smiles,
                    labels=batch_labels,
                    model_bundle=model_bundle,
                    shap_top_k=args.shap_top_k,
                    max_trees=args.max_trees,
                    class_index=args.class_index,
                    require_forest_correct=args.require_forest_correct,
                )

                for batch_offset, payload in enumerate(batch_results):
                    sample_index = pending_indices[batch_offset]
                    payload["bundle_path"] = str(bundle_path)
                    payload["input_mode"] = "tdc_split"
                    payload["task"] = task
                    payload["split"] = split
                    payload["sample_index"] = sample_index
                    payload["reasoning_schema"] = build_reasoning_schema(payload)

                    output_path = resolve_output_path(save_args, payload, bundle_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

                    split_saved += 1
                    split_selected_tree_total += len(payload["selected_trees"])

                progress.update(len(pending_indices))
        finally:
            progress.close()

        total_saved += split_saved
        total_skipped_existing += split_skipped_existing
        summary_rows.append(
            {
                "split": split,
                "num_samples": len(dataset.smiles),
                "saved": split_saved,
                "skipped_existing": split_skipped_existing,
                "mean_selected_trees_per_saved_sample": (
                    float(split_selected_tree_total / split_saved) if split_saved else 0.0
                ),
            }
        )

    print(
        json.dumps(
            {
                "task": task,
                "experiment_name": experiment_name,
                "feature_set": feature_set_name,
                "output_dir": str(output_dir),
                "total_saved": total_saved,
                "total_skipped_existing": total_skipped_existing,
                "splits": summary_rows,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
