#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD
from figs_pipeline import (
    build_training_experiment_name,
    build_feature_source_bundle,
    default_feature_config_paths,
    infer_experiment_name_from_params_json,
    resolve_experiment_output_dir,
    train_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a FIGS classifier on the TDC JSONL train split and report valid metrics."
    )
    parser.add_argument("--task", required=True, help="TDC task name")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory of TDC JSONL splits")
    parser.add_argument(
        "--feature-config",
        dest="feature_configs",
        action="append",
        default=None,
        help="Path to a feature-source JSON config. Can be repeated to concatenate multiple sources.",
    )
    parser.add_argument("--train-split", default="train", help="Split used to fit the model")
    parser.add_argument("--valid-split", default="valid", help="Split used for validation reporting")
    parser.add_argument("--smiles-key", default=DEFAULT_SMILES_FIELD, help="SMILES field name in the JSONL files")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_FIELD, help="Label field name in the JSONL files")
    parser.add_argument(
        "--params-json",
        default=None,
        help="JSON file produced by batch_tune_random_forest.py or another FIGS tuning run",
    )
    parser.add_argument("--seed", type=int, default=0, help="FIGS random_state for the final saved model")
    parser.add_argument("--rf-jobs", type=int, default=1, help="Unused by FIGS; kept for CLI compatibility")
    parser.add_argument("--scale-features", dest="scale_features", action="store_true", help="Apply standardization after median imputation")
    parser.add_argument("--no-scale-features", dest="scale_features", action="store_false", help="Disable feature scaling")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "results"), help="Root directory for experiment artifacts")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment subdirectory name under train/tree/results. If omitted, reuse the params-json experiment when possible.",
    )
    parser.set_defaults(scale_features=True)
    return parser.parse_args()


def load_training_params(params_json_path: str | None) -> dict[str, object]:
    if params_json_path is None:
        return {
            "max_rules": 12,
            "max_trees": 4,
            "min_impurity_decrease": 0.0,
            "max_features": "sqrt",
            "max_depth": None,
        }
    with Path(params_json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "best_params" in payload:
        return dict(payload["best_params"])
    return dict(payload)


def load_feature_config_paths_from_params_json(params_json_path: str | None) -> list[str] | None:
    if params_json_path is None:
        return None
    with Path(params_json_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config_paths = payload.get("feature_config_paths")
    if not isinstance(config_paths, list) or not config_paths:
        return None
    return [str(path) for path in config_paths]


def main() -> int:
    args = parse_args()
    feature_config_paths = (
        args.feature_configs
        or load_feature_config_paths_from_params_json(args.params_json)
        or default_feature_config_paths(THIS_DIR)
    )
    feature_bundle = build_feature_source_bundle(feature_config_paths)
    default_experiment_name = infer_experiment_name_from_params_json(args.params_json, base_dir=THIS_DIR)
    if default_experiment_name is None:
        default_experiment_name = build_training_experiment_name(
            feature_set_name=str(feature_bundle["feature_set_name"]),
            final_seed=args.seed,
        )
    experiment_dir = resolve_experiment_output_dir(
        base_dir=THIS_DIR,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        default_experiment_name=default_experiment_name,
    )
    summary = train_task(
        task=args.task,
        best_params=load_training_params(args.params_json),
        feature_bundle=feature_bundle,
        dataset_root=args.data_root,
        train_split_name=args.train_split,
        valid_split_name=args.valid_split,
        smiles_key=args.smiles_key,
        label_key=args.label_key,
        final_seed=args.seed,
        rf_jobs=args.rf_jobs,
        scale_features=args.scale_features,
        output_dir=experiment_dir,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
