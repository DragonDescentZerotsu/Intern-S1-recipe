#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from rf_pipeline import predict_with_model_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved RF model bundle and run predictions for one or more SMILES."
    )
    parser.add_argument("--bundle", default=None, help="Path to model_bundle.pkl")
    parser.add_argument("--bundle-root", default=str(THIS_DIR / "bundles"), help="Root directory for exported bundles")
    parser.add_argument("--experiment-name", default=None, help="Experiment name under bundle-root")
    parser.add_argument("--task", default=None, help="Task name under bundle-root/<experiment>")
    parser.add_argument("--feature-set", default=None, help="Feature-set directory name. If omitted, infer when unique.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", default=None, help="Predict a single SMILES string")
    input_group.add_argument("--smiles-file", default=None, help="Text file containing one SMILES per line")
    input_group.add_argument("--input-csv", default=None, help="CSV file containing a SMILES column")

    parser.add_argument("--smiles-column", default="smiles", help="Column name to read from --input-csv")
    parser.add_argument("--output-csv", default=None, help="Optional path to write predictions as CSV")
    parser.add_argument("--prediction-column", default="rf_prediction", help="Prediction column name in the output")
    parser.add_argument("--score-column", default="rf_score", help="Score column name in the output")
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


def load_input_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.smiles is not None:
        return pd.DataFrame({args.smiles_column: [args.smiles]})

    if args.smiles_file is not None:
        smiles_path = Path(args.smiles_file)
        smiles_values = [
            line.strip()
            for line in smiles_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not smiles_values:
            raise ValueError(f"No SMILES found in {smiles_path}")
        return pd.DataFrame({args.smiles_column: smiles_values})

    input_df = pd.read_csv(args.input_csv)
    if args.smiles_column not in input_df.columns:
        raise KeyError(f"Input CSV {args.input_csv} is missing column {args.smiles_column!r}")
    return input_df


def main() -> int:
    args = parse_args()
    bundle_path = resolve_bundle_path(args)
    input_df = load_input_frame(args)
    smiles_list = input_df[args.smiles_column].astype(str).tolist()

    predictions_df = predict_with_model_bundle(
        smiles_list=smiles_list,
        bundle_path=bundle_path,
    )

    output_df = input_df.copy()
    if args.prediction_column in output_df.columns:
        raise ValueError(f"Output column already exists: {args.prediction_column}")
    if args.score_column in output_df.columns:
        raise ValueError(f"Output column already exists: {args.score_column}")
    output_df[args.prediction_column] = predictions_df["prediction"].tolist()
    output_df[args.score_column] = predictions_df["score"].tolist()

    if args.output_csv:
        output_path = Path(args.output_csv)
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        else:
            output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_csv(sys.stdout, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
