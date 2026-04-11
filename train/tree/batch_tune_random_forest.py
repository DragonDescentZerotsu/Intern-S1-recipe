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

from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, list_tasks
from figs_pipeline import (
    build_feature_source_bundle,
    build_multi_scheme_experiment_name,
    build_tuning_experiment_name,
    default_feature_config_path_groups,
    parse_seed_list,
    resolve_experiment_output_dir,
    save_json,
    tune_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune FIGS models for TDC JSONL tasks and write a summary table."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory of TDC JSONL splits")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task names. Defaults to all tasks found under the train split.",
    )
    parser.add_argument(
        "--feature-config",
        dest="feature_configs",
        action="append",
        default=None,
        help="Path to a feature-source JSON config. Can be repeated to concatenate multiple sources.",
    )
    parser.add_argument("--train-split", default="train", help="Split used to fit the model")
    parser.add_argument(
        "--valid-split",
        default="valid",
        help="Split used for validation reporting; in 'valid' mode it is also used for model selection",
    )
    parser.add_argument("--smiles-key", default=DEFAULT_SMILES_FIELD, help="SMILES field name in the JSONL files")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_FIELD, help="Label field name in the JSONL files")
    parser.add_argument("--n-iter", type=int, default=40, help="Number of sampled hyperparameter candidates per task")
    parser.add_argument("--search-seed", type=int, default=0, help="Seed for ParameterSampler")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4", help="Comma-separated FIGS random_state seeds")
    parser.add_argument("--selection-metric", choices=["macro_f1", "roc_auc"], default="macro_f1")
    parser.add_argument(
        "--selection-mode",
        choices=["valid", "train_cv_5fold"],
        default="train_cv_5fold",
        help=(
            "How to choose hyperparameters. "
            "'valid' keeps the current train->valid selection flow; "
            "'train_cv_5fold' tunes only on train with 5-fold CV, then retrains on full train and reports valid."
        ),
    )
    parser.add_argument("--rf-jobs", type=int, default=1, help="Unused by FIGS; kept for CLI compatibility")
    parser.add_argument("--scale-features", dest="scale_features", action="store_true", help="Apply standardization after median imputation")
    parser.add_argument("--no-scale-features", dest="scale_features", action="store_false", help="Disable feature scaling")
    parser.add_argument("--output-dir", default=str(THIS_DIR / "results"), help="Root directory for experiment artifacts")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment subdirectory name under train/tree/results. Defaults to a name derived from the tuning setup.",
    )
    parser.add_argument("--summary-name", default="all_tasks_tuning_summary", help="Base filename for aggregate outputs")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop immediately if one task fails")
    parser.set_defaults(scale_features=True)
    return parser.parse_args()


def parse_task_list(task_text: str | None, data_root: str) -> list[str]:
    if task_text is None:
        return list_tasks(data_root)
    tasks = [item.strip() for item in task_text.split(",") if item.strip()]
    if not tasks:
        raise ValueError("If --tasks is provided it must contain at least one task name")
    return tasks


def resolve_feature_bundles(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.feature_configs:
        return [build_feature_source_bundle(args.feature_configs)]
    return [
        build_feature_source_bundle(config_paths)
        for config_paths in default_feature_config_path_groups(THIS_DIR)
    ]


def main() -> int:
    args = parse_args()
    tasks = parse_task_list(args.tasks, args.data_root)
    feature_bundles = resolve_feature_bundles(args)
    eval_seeds = parse_seed_list(args.eval_seeds)
    if len(feature_bundles) == 1:
        default_experiment_name = build_tuning_experiment_name(
            feature_set_name=str(feature_bundles[0]["feature_set_name"]),
            selection_mode=args.selection_mode,
            n_iter=args.n_iter,
            search_seed=args.search_seed,
            eval_seeds=eval_seeds,
        )
    else:
        default_experiment_name = build_multi_scheme_experiment_name(
            selection_mode=args.selection_mode,
            n_iter=args.n_iter,
            search_seed=args.search_seed,
            eval_seeds=eval_seeds,
        )
    experiment_dir = resolve_experiment_output_dir(
        base_dir=THIS_DIR,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        default_experiment_name=default_experiment_name,
    )

    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for feature_bundle in feature_bundles:
        feature_set_name = str(feature_bundle["feature_set_name"])
        for task in tasks:
            print(f"Running FIGS tuning for task={task} feature_set={feature_set_name}")
            try:
                result = tune_task(
                    task=task,
                    feature_bundle=feature_bundle,
                    dataset_root=args.data_root,
                    train_split_name=args.train_split,
                    valid_split_name=args.valid_split,
                    smiles_key=args.smiles_key,
                    label_key=args.label_key,
                    n_iter=args.n_iter,
                    search_seed=args.search_seed,
                    eval_seeds=eval_seeds,
                    selection_metric=args.selection_metric,
                    selection_mode=args.selection_mode,
                    rf_jobs=args.rf_jobs,
                    scale_features=args.scale_features,
                    output_dir=experiment_dir,
                )
                summary_rows.append(
                    {
                        "task": task,
                        "model_family": result["model_family"],
                        "feature_set_name": result["feature_set_name"],
                        "selection_metric": result["selection_metric"],
                        "selection_mode": result["selection_mode"],
                        "mean_valid_macro_f1": result["best_metrics"]["mean_valid_macro_f1"],
                        "std_valid_macro_f1": result["best_metrics"]["std_valid_macro_f1"],
                        "mean_valid_roc_auc": result["best_metrics"]["mean_valid_roc_auc"],
                        "std_valid_roc_auc": result["best_metrics"]["std_valid_roc_auc"],
                        "mean_train_macro_f1": result["best_metrics"]["mean_train_macro_f1"],
                        "mean_train_roc_auc": result["best_metrics"]["mean_train_roc_auc"],
                        "final_valid_macro_f1": result.get("final_model_metrics", {}).get("valid", {}).get("macro_f1"),
                        "final_valid_roc_auc": result.get("final_model_metrics", {}).get("valid", {}).get("roc_auc"),
                        "num_surviving_features": result["num_surviving_features"],
                        "best_params_json": result["artifacts"]["best_params_json"],
                    }
                )
            except Exception as exc:
                failures.append({"task": task, "feature_set_name": feature_set_name, "error": str(exc)})
                print(f"Task {task} with feature_set={feature_set_name} failed: {exc}")
                if args.stop_on_error:
                    raise

    experiment_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["task", "final_valid_macro_f1", "mean_valid_macro_f1", "final_valid_roc_auc"],
            ascending=[True, False, False, False],
        )
    summary_csv = experiment_dir / f"{args.summary_name}.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_payload = {
        "tasks": tasks,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir.resolve()),
        "model_family": "figs",
        "feature_sets": [
            {
                "feature_set_name": bundle["feature_set_name"],
                "feature_config_paths": bundle["config_paths"],
            }
            for bundle in feature_bundles
        ],
        "selection_metric": args.selection_metric,
        "selection_mode": args.selection_mode,
        "n_iter": args.n_iter,
        "search_seed": args.search_seed,
        "eval_seeds": eval_seeds,
        "summary_csv": str(summary_csv.resolve()),
        "num_successful_tasks": len(summary_rows),
        "num_failed_tasks": len(failures),
        "failures": failures,
    }
    summary_json = experiment_dir / f"{args.summary_name}.json"
    save_json(summary_json, summary_payload)

    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
