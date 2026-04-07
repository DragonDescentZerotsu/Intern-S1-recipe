from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path


def configure_single_thread_runtime() -> None:
    thread_limits = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key, value in thread_limits.items():
        os.environ.setdefault(key, value)


configure_single_thread_runtime()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, StratifiedKFold

try:
    from .data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, load_tdc_split
    from .feature_sources import build_composite_feature_source, load_feature_specs_from_paths
    from .metrics import compute_binary_classification_metrics
    from .preprocessing import prepare_feature_matrices, transform_feature_frame
except ImportError:
    from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, load_tdc_split
    from feature_sources import build_composite_feature_source, load_feature_specs_from_paths
    from metrics import compute_binary_classification_metrics
    from preprocessing import prepare_feature_matrices, transform_feature_frame

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable


RESULTS_DIRNAME = "results"


def default_feature_config_paths(base_dir: str | Path) -> list[str]:
    return [str(Path(base_dir) / "configs" / "fg_top_level_features.json")]


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("eval_seeds must contain at least one integer")
    return seeds


def get_parameter_space() -> dict[str, list[object]]:
    return {
        "n_estimators": [100, 150, 200, 300, 400, 500, 700],
        "max_depth": [None, 4, 8, 10, 20, 30, 40],
        "min_samples_split": [2, 5, 10, 20, 50, 100],
        "min_samples_leaf": [1, 2, 3, 4, 5, 10],
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }


def infer_feature_set_name(loaded_configs: list[dict[str, object]], feature_source_name: str) -> str:
    configured_names = [
        str(config["feature_set_name"])
        for config in loaded_configs
        if config.get("feature_set_name")
    ]
    if configured_names:
        return "+".join(configured_names)
    return feature_source_name.replace("/", "_")


def normalize_experiment_name(experiment_name: str) -> str:
    normalized = str(experiment_name).strip().strip("/\\")
    if not normalized:
        raise ValueError("experiment_name must not be empty")
    return normalized.replace("\\", "__").replace("/", "__").replace(" ", "_")


def format_seed_suffix(seeds: list[int]) -> str:
    return "-".join(str(seed) for seed in seeds)


def build_tuning_experiment_name(
    *,
    feature_set_name: str,
    selection_mode: str,
    n_iter: int,
    search_seed: int,
    eval_seeds: list[int],
) -> str:
    return normalize_experiment_name(
        f"{feature_set_name}__{selection_mode}__n{n_iter}__search{search_seed}__eval{format_seed_suffix(eval_seeds)}"
    )


def build_training_experiment_name(*, feature_set_name: str, final_seed: int) -> str:
    return normalize_experiment_name(f"{feature_set_name}__train__seed{final_seed}")


def infer_experiment_name_from_params_json(
    params_json_path: str | Path | None,
    *,
    base_dir: str | Path,
) -> str | None:
    if not params_json_path:
        return None

    params_path = Path(params_json_path).expanduser()
    if not params_path.is_absolute():
        params_path = (Path.cwd() / params_path).resolve()
    else:
        params_path = params_path.resolve()

    results_root = Path(base_dir).resolve() / RESULTS_DIRNAME
    try:
        relative_path = params_path.relative_to(results_root)
    except ValueError:
        return None

    if len(relative_path.parts) < 4:
        return None
    return relative_path.parts[0]


def resolve_experiment_output_dir(
    *,
    base_dir: str | Path,
    output_dir: str | Path,
    experiment_name: str | None,
    default_experiment_name: str,
) -> Path:
    base_dir = Path(base_dir).resolve()
    results_root = base_dir / RESULTS_DIRNAME

    requested_output_dir = Path(output_dir).expanduser()
    if not requested_output_dir.is_absolute():
        requested_output_dir = (Path.cwd() / requested_output_dir).resolve()
    else:
        requested_output_dir = requested_output_dir.resolve()

    inferred_experiment_name = None
    if requested_output_dir.parent == base_dir and requested_output_dir.name.startswith(f"{RESULTS_DIRNAME}_"):
        inferred_experiment_name = requested_output_dir.name[len(RESULTS_DIRNAME) + 1 :]
    elif requested_output_dir.parent == results_root:
        inferred_experiment_name = requested_output_dir.name
    elif requested_output_dir != results_root and requested_output_dir.name != RESULTS_DIRNAME:
        inferred_experiment_name = requested_output_dir.name

    final_experiment_name = experiment_name or inferred_experiment_name or default_experiment_name
    return results_root / normalize_experiment_name(final_experiment_name)


def validate_binary_labels(
    labels: list[int],
    *,
    split_name: str,
    require_both_classes: bool,
) -> list[int]:
    integer_labels = [int(label) for label in labels]
    label_set = set(integer_labels)
    if not label_set.issubset({0, 1}):
        raise ValueError(
            f"{split_name} split must contain only binary labels 0/1. "
            f"Observed labels: {sorted(label_set)}"
        )
    if require_both_classes and len(label_set) < 2:
        raise ValueError(f"{split_name} split contains only one class: {sorted(label_set)}")
    return integer_labels


def get_metric_sort_value(value: float, *, descending: bool) -> float:
    if math.isnan(value):
        return float("-inf") if descending else float("inf")
    return value


def save_json(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def build_feature_source_bundle(
    config_paths: list[str | Path],
) -> dict[str, object]:
    feature_specs, loaded_configs = load_feature_specs_from_paths(list(config_paths))
    feature_source = build_composite_feature_source(feature_specs)
    feature_set_name = infer_feature_set_name(loaded_configs, feature_source.name)
    return {
        "feature_source": feature_source,
        "feature_set_name": feature_set_name,
        "loaded_configs": loaded_configs,
        "config_paths": [str(Path(path).resolve()) for path in config_paths],
    }


def load_model_bundle(bundle_path: str | Path) -> dict[str, object]:
    bundle_path = Path(bundle_path)
    with bundle_path.open("rb") as handle:
        model_bundle = pickle.load(handle)
    if not isinstance(model_bundle, dict):
        raise ValueError(f"Bundle at {bundle_path} did not contain a dict payload")
    required_keys = {"feature_config_paths", "preprocessor", "model"}
    missing_keys = sorted(required_keys - set(model_bundle))
    if missing_keys:
        raise KeyError(f"Bundle at {bundle_path} is missing required keys: {missing_keys}")
    return model_bundle


def build_feature_source_for_model_bundle(model_bundle: dict[str, object]):
    feature_config_paths = model_bundle.get("feature_config_paths")
    if not isinstance(feature_config_paths, list) or not feature_config_paths:
        raise ValueError("Model bundle must contain a non-empty 'feature_config_paths' list")
    feature_bundle = build_feature_source_bundle(feature_config_paths)
    return feature_bundle["feature_source"]


def predict_with_model_bundle(
    *,
    smiles_list: list[str],
    model_bundle: dict[str, object] | None = None,
    bundle_path: str | Path | None = None,
) -> pd.DataFrame:
    if model_bundle is None:
        if bundle_path is None:
            raise ValueError("Either model_bundle or bundle_path must be provided")
        model_bundle = load_model_bundle(bundle_path)

    feature_source = build_feature_source_for_model_bundle(model_bundle)
    feature_frame = feature_source.load(smiles_list)
    _, transformed_frame = transform_feature_frame(feature_frame, model_bundle["preprocessor"])

    model = model_bundle["model"]
    x_matrix = transformed_frame.to_numpy()
    predictions = model.predict(x_matrix)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_matrix)[:, 1]
    else:
        scores = predictions

    return pd.DataFrame(
        {
            "smiles": [str(smiles) for smiles in smiles_list],
            "prediction": predictions,
            "score": scores,
        }
    )


def load_task_feature_matrices(
    *,
    task: str,
    dataset_root: str | Path = DEFAULT_DATA_ROOT,
    train_split_name: str = "train",
    valid_split_name: str = "valid",
    smiles_key: str = DEFAULT_SMILES_FIELD,
    label_key: str = DEFAULT_LABEL_FIELD,
    feature_source,
    scale_features: bool = False,
) -> dict[str, object]:
    train_split = load_tdc_split(
        task,
        train_split_name,
        data_root=dataset_root,
        smiles_field=smiles_key,
        label_field=label_key,
    )
    valid_split = load_tdc_split(
        task,
        valid_split_name,
        data_root=dataset_root,
        smiles_field=smiles_key,
        label_field=label_key,
    )

    y_train = validate_binary_labels(train_split.labels, split_name=train_split_name, require_both_classes=True)
    y_valid = validate_binary_labels(valid_split.labels, split_name=valid_split_name, require_both_classes=False)

    train_feature_frame = feature_source.load(train_split.smiles)
    valid_feature_frame = feature_source.load(valid_split.smiles)
    x_train, x_valid, surviving_columns, preprocessor = prepare_feature_matrices(
        train_feature_frame,
        valid_feature_frame,
        scale_features=scale_features,
        return_preprocessor=True,
    )
    return {
        "train_split": train_split,
        "valid_split": valid_split,
        "y_train": y_train,
        "y_valid": y_valid,
        "train_feature_frame": train_feature_frame,
        "valid_feature_frame": valid_feature_frame,
        "x_train": x_train,
        "x_valid": x_valid,
        "surviving_columns": surviving_columns,
        "preprocessor": preprocessor,
    }


def evaluate_candidate(params, x_train, y_train, x_valid, y_valid, seeds: list[int], rf_jobs: int) -> dict[str, float]:
    valid_macro_f1_scores = []
    valid_roc_auc_scores = []
    train_macro_f1_scores = []
    train_roc_auc_scores = []

    for seed in seeds:
        model = RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=rf_jobs,
        )
        model.fit(x_train, y_train)

        train_scores = model.predict_proba(x_train)[:, 1]
        valid_scores = model.predict_proba(x_valid)[:, 1]
        train_predictions = model.predict(x_train)
        valid_predictions = model.predict(x_valid)

        train_metrics = compute_binary_classification_metrics(y_train, train_predictions, train_scores)
        valid_metrics = compute_binary_classification_metrics(y_valid, valid_predictions, valid_scores)

        train_macro_f1_scores.append(train_metrics["macro_f1"])
        train_roc_auc_scores.append(train_metrics["roc_auc"])
        valid_macro_f1_scores.append(valid_metrics["macro_f1"])
        valid_roc_auc_scores.append(valid_metrics["roc_auc"])

    return {
        "mean_train_macro_f1": float(np.nanmean(train_macro_f1_scores)),
        "std_train_macro_f1": float(np.nanstd(train_macro_f1_scores)),
        "mean_valid_macro_f1": float(np.nanmean(valid_macro_f1_scores)),
        "std_valid_macro_f1": float(np.nanstd(valid_macro_f1_scores)),
        "mean_train_roc_auc": float(np.nanmean(train_roc_auc_scores)),
        "std_train_roc_auc": float(np.nanstd(train_roc_auc_scores)),
        "mean_valid_roc_auc": float(np.nanmean(valid_roc_auc_scores)),
        "std_valid_roc_auc": float(np.nanstd(valid_roc_auc_scores)),
    }


def evaluate_candidate_with_train_cv(
    params,
    train_feature_frame: pd.DataFrame,
    y_train: list[int],
    *,
    seeds: list[int],
    rf_jobs: int,
    scale_features: bool,
    cv_n_splits: int,
    cv_seed: int,
) -> dict[str, float]:
    y_train_array = np.asarray(y_train, dtype=int)
    class_counts = np.bincount(y_train_array, minlength=2)
    min_class_count = int(class_counts[class_counts > 0].min())
    if min_class_count < cv_n_splits:
        raise ValueError(
            f"Cannot run {cv_n_splits}-fold CV because the smallest class in train has only "
            f"{min_class_count} samples."
        )

    splitter = StratifiedKFold(
        n_splits=cv_n_splits,
        shuffle=True,
        random_state=cv_seed,
    )

    valid_macro_f1_scores = []
    valid_roc_auc_scores = []
    train_macro_f1_scores = []
    train_roc_auc_scores = []

    for fold_train_idx, fold_valid_idx in splitter.split(train_feature_frame, y_train_array):
        fold_train_df = train_feature_frame.iloc[fold_train_idx]
        fold_valid_df = train_feature_frame.iloc[fold_valid_idx]
        x_fold_train, x_fold_valid, _ = prepare_feature_matrices(
            fold_train_df,
            fold_valid_df,
            scale_features=scale_features,
        )
        y_fold_train = y_train_array[fold_train_idx]
        y_fold_valid = y_train_array[fold_valid_idx]

        for seed in seeds:
            model = RandomForestClassifier(
                **params,
                random_state=seed,
                n_jobs=rf_jobs,
            )
            model.fit(x_fold_train, y_fold_train)

            train_scores = model.predict_proba(x_fold_train)[:, 1]
            valid_scores = model.predict_proba(x_fold_valid)[:, 1]
            train_predictions = model.predict(x_fold_train)
            valid_predictions = model.predict(x_fold_valid)

            train_metrics = compute_binary_classification_metrics(y_fold_train, train_predictions, train_scores)
            valid_metrics = compute_binary_classification_metrics(y_fold_valid, valid_predictions, valid_scores)

            train_macro_f1_scores.append(train_metrics["macro_f1"])
            train_roc_auc_scores.append(train_metrics["roc_auc"])
            valid_macro_f1_scores.append(valid_metrics["macro_f1"])
            valid_roc_auc_scores.append(valid_metrics["roc_auc"])

    return {
        "mean_train_macro_f1": float(np.nanmean(train_macro_f1_scores)),
        "std_train_macro_f1": float(np.nanstd(train_macro_f1_scores)),
        "mean_valid_macro_f1": float(np.nanmean(valid_macro_f1_scores)),
        "std_valid_macro_f1": float(np.nanstd(valid_macro_f1_scores)),
        "mean_train_roc_auc": float(np.nanmean(train_roc_auc_scores)),
        "std_train_roc_auc": float(np.nanstd(train_roc_auc_scores)),
        "mean_valid_roc_auc": float(np.nanmean(valid_roc_auc_scores)),
        "std_valid_roc_auc": float(np.nanstd(valid_roc_auc_scores)),
    }


def extract_best_params(record: dict[str, object]) -> dict[str, object]:
    keys = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "bootstrap",
        "criterion",
        "max_features",
        "class_weight",
    ]
    return {key: record[key] for key in keys}


def tune_task(
    *,
    task: str,
    feature_bundle: dict[str, object],
    dataset_root: str | Path = DEFAULT_DATA_ROOT,
    train_split_name: str = "train",
    valid_split_name: str = "valid",
    smiles_key: str = DEFAULT_SMILES_FIELD,
    label_key: str = DEFAULT_LABEL_FIELD,
    n_iter: int = 40,
    search_seed: int = 0,
    eval_seeds: list[int] | None = None,
    selection_metric: str = "macro_f1",
    selection_mode: str = "valid",
    rf_jobs: int = 1,
    scale_features: bool = False,
    output_dir: str | Path = "train/tree/results/default_experiment",
) -> dict[str, object]:
    if eval_seeds is None:
        eval_seeds = [0, 1, 2, 3, 4]

    matrices = load_task_feature_matrices(
        task=task,
        dataset_root=dataset_root,
        train_split_name=train_split_name,
        valid_split_name=valid_split_name,
        smiles_key=smiles_key,
        label_key=label_key,
        feature_source=feature_bundle["feature_source"],
        scale_features=scale_features,
    )

    x_train = matrices["x_train"]
    y_train = matrices["y_train"]
    x_valid = matrices["x_valid"]
    y_valid = matrices["y_valid"]
    train_feature_frame = matrices["train_feature_frame"]
    surviving_columns = matrices["surviving_columns"]

    if selection_mode not in {"valid", "train_cv_5fold"}:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    candidate_params = list(
        ParameterSampler(
            get_parameter_space(),
            n_iter=n_iter,
            random_state=search_seed,
        )
    )
    selection_mean_key = f"mean_valid_{selection_metric}"
    selection_std_key = f"std_valid_{selection_metric}"
    results = []
    best_record = None

    for candidate_index, params in enumerate(
        tqdm(candidate_params, total=len(candidate_params), desc=f"RF tuning {task}"),
        start=1,
    ):
        if selection_mode == "train_cv_5fold":
            metrics = evaluate_candidate_with_train_cv(
                params,
                train_feature_frame,
                y_train,
                seeds=eval_seeds,
                rf_jobs=rf_jobs,
                scale_features=scale_features,
                cv_n_splits=5,
                cv_seed=search_seed,
            )
        else:
            metrics = evaluate_candidate(
                params,
                x_train,
                y_train,
                x_valid,
                y_valid,
                eval_seeds,
                rf_jobs,
            )
        record = {
            "candidate_index": candidate_index,
            **params,
            **metrics,
        }
        results.append(record)
        if best_record is None:
            best_record = record
            continue

        record_mean = get_metric_sort_value(record[selection_mean_key], descending=True)
        best_mean = get_metric_sort_value(best_record[selection_mean_key], descending=True)
        record_std = get_metric_sort_value(record[selection_std_key], descending=False)
        best_std = get_metric_sort_value(best_record[selection_std_key], descending=False)
        if (record_mean > best_mean) or (record_mean == best_mean and record_std < best_std):
            best_record = record

    if best_record is None:
        raise RuntimeError(f"No hyperparameter candidates were evaluated for task {task}")

    experiment_dir = Path(output_dir)
    task_output_dir = experiment_dir / task / str(feature_bundle["feature_set_name"])
    task_output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results).sort_values(
        by=[selection_mean_key, selection_std_key, "mean_valid_roc_auc"],
        ascending=[False, True, False],
    )
    results_csv = task_output_dir / "rf_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    best_params = extract_best_params(best_record)
    final_valid_report = None
    final_eval_seed = None
    if selection_mode == "train_cv_5fold":
        final_eval_seed = eval_seeds[0]
        final_model = RandomForestClassifier(
            **best_params,
            random_state=final_eval_seed,
            n_jobs=rf_jobs,
        )
        final_model.fit(x_train, y_train)

        train_scores = final_model.predict_proba(x_train)[:, 1]
        valid_scores = final_model.predict_proba(x_valid)[:, 1]
        train_predictions = final_model.predict(x_train)
        valid_predictions = final_model.predict(x_valid)
        final_valid_report = {
            "train": compute_binary_classification_metrics(y_train, train_predictions, train_scores),
            "valid": compute_binary_classification_metrics(y_valid, valid_predictions, valid_scores),
        }

    summary = {
        "task": task,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": train_split_name,
        "valid_split": valid_split_name,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "search_seed": search_seed,
        "eval_seeds": eval_seeds,
        "feature_set_name": feature_bundle["feature_set_name"],
        "feature_config_paths": feature_bundle["config_paths"],
        "loaded_feature_configs": feature_bundle["loaded_configs"],
        "feature_source": feature_bundle["feature_source"].describe(),
        "num_surviving_features": len(surviving_columns),
        "surviving_feature_names": surviving_columns,
        "best_params": best_params,
        "best_metrics": {
            key: best_record[key]
            for key in [
                "mean_train_roc_auc",
                "std_train_roc_auc",
                "mean_valid_roc_auc",
                "std_valid_roc_auc",
                "mean_train_macro_f1",
                "std_train_macro_f1",
                "mean_valid_macro_f1",
                "std_valid_macro_f1",
            ]
        },
        "artifacts": {
            "search_results_csv": str(results_csv.resolve()),
        },
    }
    if selection_mode == "train_cv_5fold":
        summary["cv_num_folds"] = 5
    if final_valid_report is not None:
        summary["final_eval_seed"] = final_eval_seed
        summary["final_model_metrics"] = final_valid_report
    best_json = task_output_dir / "best_params.json"
    save_json(best_json, summary)
    summary["artifacts"]["best_params_json"] = str(best_json.resolve())
    save_json(best_json, summary)
    return summary


def train_task(
    *,
    task: str,
    best_params: dict[str, object],
    feature_bundle: dict[str, object],
    dataset_root: str | Path = DEFAULT_DATA_ROOT,
    train_split_name: str = "train",
    valid_split_name: str = "valid",
    smiles_key: str = DEFAULT_SMILES_FIELD,
    label_key: str = DEFAULT_LABEL_FIELD,
    final_seed: int = 0,
    rf_jobs: int = 1,
    scale_features: bool = False,
    output_dir: str | Path = "train/tree/results/default_experiment",
) -> dict[str, object]:
    matrices = load_task_feature_matrices(
        task=task,
        dataset_root=dataset_root,
        train_split_name=train_split_name,
        valid_split_name=valid_split_name,
        smiles_key=smiles_key,
        label_key=label_key,
        feature_source=feature_bundle["feature_source"],
        scale_features=scale_features,
    )

    x_train = matrices["x_train"]
    y_train = matrices["y_train"]
    x_valid = matrices["x_valid"]
    y_valid = matrices["y_valid"]
    train_split = matrices["train_split"]
    valid_split = matrices["valid_split"]
    surviving_columns = matrices["surviving_columns"]
    preprocessor = matrices["preprocessor"]

    model = RandomForestClassifier(
        **best_params,
        random_state=final_seed,
        n_jobs=rf_jobs,
    )
    model.fit(x_train, y_train)

    train_scores = model.predict_proba(x_train)[:, 1]
    valid_scores = model.predict_proba(x_valid)[:, 1]
    train_predictions = model.predict(x_train)
    valid_predictions = model.predict(x_valid)

    final_train_metrics = compute_binary_classification_metrics(y_train, train_predictions, train_scores)
    final_valid_metrics = compute_binary_classification_metrics(y_valid, valid_predictions, valid_scores)

    experiment_dir = Path(output_dir)
    task_output_dir = experiment_dir / task / str(feature_bundle["feature_set_name"])
    task_output_dir.mkdir(parents=True, exist_ok=True)

    train_predictions_df = pd.DataFrame(
        {
            "smiles": train_split.smiles,
            "label": y_train,
            "prediction": train_predictions,
            "score": train_scores,
        }
    )
    valid_predictions_df = pd.DataFrame(
        {
            "smiles": valid_split.smiles,
            "label": y_valid,
            "prediction": valid_predictions,
            "score": valid_scores,
        }
    )
    train_predictions_csv = task_output_dir / "train_predictions.csv"
    valid_predictions_csv = task_output_dir / "valid_predictions.csv"
    train_predictions_df.to_csv(train_predictions_csv, index=False)
    valid_predictions_df.to_csv(valid_predictions_csv, index=False)

    model_bundle = {
        "task": task,
        "feature_set_name": feature_bundle["feature_set_name"],
        "feature_config_paths": feature_bundle["config_paths"],
        "feature_source": feature_bundle["feature_source"].describe(),
        "feature_columns": surviving_columns,
        "preprocessor": preprocessor,
        "rf_params": best_params,
        "model": model,
        "metrics": {
            "train": final_train_metrics,
            "valid": final_valid_metrics,
        },
    }
    model_bundle_path = task_output_dir / "model_bundle.pkl"
    with model_bundle_path.open("wb") as handle:
        pickle.dump(model_bundle, handle)

    summary = {
        "task": task,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": train_split_name,
        "valid_split": valid_split_name,
        "final_seed": final_seed,
        "feature_set_name": feature_bundle["feature_set_name"],
        "feature_config_paths": feature_bundle["config_paths"],
        "loaded_feature_configs": feature_bundle["loaded_configs"],
        "feature_source": feature_bundle["feature_source"].describe(),
        "num_surviving_features": len(surviving_columns),
        "surviving_feature_names": surviving_columns,
        "rf_params": best_params,
        "final_model_metrics": {
            "train": final_train_metrics,
            "valid": final_valid_metrics,
        },
        "artifacts": {
            "train_predictions_csv": str(train_predictions_csv.resolve()),
            "valid_predictions_csv": str(valid_predictions_csv.resolve()),
            "model_bundle_pkl": str(model_bundle_path.resolve()),
        },
    }
    summary_json = task_output_dir / "train_summary.json"
    save_json(summary_json, summary)
    return summary
