#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, get_tdc_split_sample
from rf_pipeline import explain_with_model_bundle, load_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain per-sample RandomForest feature contributions with TreeSHAP."
    )
    parser.add_argument("--bundle", default=None, help="Path to model_bundle.pkl")
    parser.add_argument("--bundle-root", default=str(THIS_DIR / "bundles"), help="Root directory for exported bundles")
    parser.add_argument("--experiment-name", default=None, help="Experiment name under bundle-root")
    parser.add_argument("--task", default=None, help="Task name under bundle-root/<experiment>")
    parser.add_argument("--feature-set", default=None, help="Feature-set directory name. If omitted, infer when unique.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", default=None, help="Explain a single SMILES string")
    input_group.add_argument("--smiles-file", default=None, help="Text file containing one SMILES per line")
    input_group.add_argument("--input-csv", default=None, help="CSV file containing a SMILES column")
    input_group.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample index within a TDC split. Requires --split and either --task or a bundle carrying task metadata.",
    )

    parser.add_argument("--split", default="valid", help="Dataset split used with --sample-index")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory of TDC JSONL splits")
    parser.add_argument("--smiles-column", default="smiles", help="Column name to read from --input-csv")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_FIELD, help="Label field name in the JSONL files")
    parser.add_argument("--smiles-key", default=DEFAULT_SMILES_FIELD, help="SMILES field name in the JSONL files")
    parser.add_argument(
        "--class-index",
        type=int,
        default=1,
        help="Class index to explain. Default 1 matches the RF probability score used elsewhere in train/tree.",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Number of highest-impact features to keep per sample")
    parser.add_argument("--output-json", default=None, help="Optional path to write the explanation payload as JSON")
    parser.add_argument("--output-csv", default=None, help="Optional path to write flattened feature contributions as CSV")
    return parser.parse_args()


def resolve_bundle_path(args: argparse.Namespace) -> Path:
    if args.bundle:
        bundle_path = Path(args.bundle).expanduser()
        return bundle_path.resolve() if bundle_path.is_absolute() else (Path.cwd() / bundle_path).resolve()

    if not args.experiment_name or not args.task:
        raise ValueError("Either --bundle or both --experiment-name and --task must be provided")

    task_dir = Path(args.bundle_root).expanduser()
    if not task_dir.is_absolute():
        task_dir = (Path.cwd() / task_dir).resolve()
    else:
        task_dir = task_dir.resolve()
    task_dir = task_dir / args.experiment_name / args.task

    if args.feature_set:
        bundle_path = task_dir / args.feature_set / "model_bundle.pkl"
    else:
        candidates = sorted(task_dir.glob("*/model_bundle.pkl"))
        if not candidates:
            raise FileNotFoundError(f"No bundle found under {task_dir}")
        if len(candidates) > 1:
            raise ValueError(f"Multiple feature-set bundles found under {task_dir}; please pass --feature-set")
        bundle_path = candidates[0]

    return bundle_path.resolve()


def resolve_task_for_sample(args: argparse.Namespace, bundle_path: Path) -> str:
    if args.task:
        return args.task
    bundle = load_model_bundle(bundle_path)
    task = bundle.get("task")
    if not task:
        raise ValueError("Could not infer task from model bundle; please pass --task explicitly")
    return str(task)


def load_inputs(args: argparse.Namespace, bundle_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    if args.smiles is not None:
        return [args.smiles], [{"input_mode": "smiles", "input_row_index": 0}]

    if args.smiles_file is not None:
        smiles_path = Path(args.smiles_file)
        smiles_values = [
            line.strip()
            for line in smiles_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not smiles_values:
            raise ValueError(f"No SMILES found in {smiles_path}")
        metadata_rows = [
            {"input_mode": "smiles_file", "input_row_index": row_index}
            for row_index in range(len(smiles_values))
        ]
        return smiles_values, metadata_rows

    if args.input_csv is not None:
        input_df = pd.read_csv(args.input_csv)
        if args.smiles_column not in input_df.columns:
            raise KeyError(f"Input CSV {args.input_csv} is missing column {args.smiles_column!r}")
        smiles_values = input_df[args.smiles_column].astype(str).tolist()
        metadata_rows = input_df.to_dict(orient="records")
        for row_index, metadata in enumerate(metadata_rows):
            metadata.setdefault("input_mode", "input_csv")
            metadata.setdefault("input_row_index", row_index)
        return smiles_values, metadata_rows

    task = resolve_task_for_sample(args, bundle_path)
    sample = get_tdc_split_sample(
        task=task,
        split=args.split,
        sample_index=args.sample_index,
        data_root=args.data_root,
        smiles_field=args.smiles_key,
        label_field=args.label_key,
    )
    metadata = {
        "input_mode": "tdc_split",
        "task": sample["task"],
        "split": sample["split"],
        "sample_index": sample["sample_index"],
        "label": sample["label"],
    }
    return [str(sample["smiles"])], [metadata]


def flatten_explanations(explanations: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for explanation in explanations:
        sample_columns = {
            key: value
            for key, value in explanation.items()
            if key != "features"
        }
        for rank, feature in enumerate(explanation["features"], start=1):
            rows.append(
                {
                    **sample_columns,
                    "feature_rank": rank,
                    **feature,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    bundle_path = resolve_bundle_path(args)
    smiles_list, metadata_rows = load_inputs(args, bundle_path)

    explanations = explain_with_model_bundle(
        smiles_list=smiles_list,
        bundle_path=bundle_path,
        class_index=args.class_index,
        top_k=args.top_k,
    )

    for explanation, metadata in zip(explanations, metadata_rows):
        explanation["bundle_path"] = str(bundle_path)
        explanation.update(metadata)

    if args.output_json:
        output_json_path = Path(args.output_json)
        if not output_json_path.is_absolute():
            output_json_path = (Path.cwd() / output_json_path).resolve()
        else:
            output_json_path = output_json_path.resolve()
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(explanations, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.output_csv:
        output_csv_path = Path(args.output_csv)
        if not output_csv_path.is_absolute():
            output_csv_path = (Path.cwd() / output_csv_path).resolve()
        else:
            output_csv_path = output_csv_path.resolve()
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        flatten_explanations(explanations).to_csv(output_csv_path, index=False)

    if not args.output_json and not args.output_csv:
        print(json.dumps(explanations, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
