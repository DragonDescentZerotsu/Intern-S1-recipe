from __future__ import annotations

import itertools
import os
import tempfile
from pathlib import Path


def configure_runtime() -> None:
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

    matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))


configure_runtime()

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    from imodels import FIGSClassifier
except ImportError as exc:  # pragma: no cover - handled with a clearer error later
    FIGSClassifier = None
    _FIGS_IMPORT_ERROR = exc
else:
    _FIGS_IMPORT_ERROR = None

try:
    from .metrics import compute_binary_classification_metrics
    from .preprocessing import prepare_feature_matrices
    from .rf_pipeline import (
        DEFAULT_DATA_ROOT,
        DEFAULT_LABEL_FIELD,
        DEFAULT_SMILES_FIELD,
        build_feature_source_bundle,
        default_feature_config_paths,
        format_seed_suffix,
        get_metric_sort_value,
        infer_experiment_name_from_params_json,
        load_task_feature_matrices,
        normalize_experiment_name,
        parse_seed_list,
        resolve_experiment_output_dir,
        save_json,
    )
except ImportError:
    from metrics import compute_binary_classification_metrics
    from preprocessing import prepare_feature_matrices
    from rf_pipeline import (
        DEFAULT_DATA_ROOT,
        DEFAULT_LABEL_FIELD,
        DEFAULT_SMILES_FIELD,
        build_feature_source_bundle,
        default_feature_config_paths,
        format_seed_suffix,
        get_metric_sort_value,
        infer_experiment_name_from_params_json,
        load_task_feature_matrices,
        normalize_experiment_name,
        parse_seed_list,
        resolve_experiment_output_dir,
        save_json,
    )


def require_figs() -> None:
    if FIGSClassifier is None:
        raise ImportError(
            "FIGSClassifier is unavailable because the 'imodels' package could not be imported. "
            "Please run this pipeline in the vllm conda environment."
        ) from _FIGS_IMPORT_ERROR


def default_feature_config_path_groups(base_dir: str | Path) -> list[list[str]]:
    configs_dir = Path(base_dir) / "configs"
    return [
        [str(configs_dir / "rdkit_descriptors_and_pka_easy_to_NLP_Lv1_features.json")],
        [str(configs_dir / "fg_top_level_features.json")],
        [str(configs_dir / "fg_top_level_plus_rdkit_descriptors_and_pka_easy_to_NLP_Lv1_features.json")],
    ]


def get_parameter_space() -> dict[str, list[object]]:
    return {
        "max_rules": [4, 6, 8, 12, 16, 24, 32, 48],
        "max_trees": [1, 2, 3, 4, 6, 8, None],
        "min_impurity_decrease": [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "max_features": [None, "sqrt", "log2"],
        "max_depth": [None, 2, 3, 4, 5, 6, 8, 10],
    }


def list_valid_parameter_configs() -> list[dict[str, object]]:
    parameter_space = get_parameter_space()
    parameter_names = list(parameter_space)
    candidates = []
    for values in itertools.product(*(parameter_space[name] for name in parameter_names)):
        params = dict(zip(parameter_names, values))
        max_rules = int(params["max_rules"])
        max_trees = params["max_trees"]
        if max_trees is not None and int(max_trees) > max_rules:
            continue
        candidates.append(params)
    return candidates


def sample_parameter_configs(n_iter: int, search_seed: int) -> list[dict[str, object]]:
    all_candidates = list_valid_parameter_configs()
    if not all_candidates:
        raise RuntimeError("No valid FIGS hyperparameter candidates were generated")

    if n_iter >= len(all_candidates):
        return all_candidates

    random_state = np.random.RandomState(search_seed)
    selected_indices = random_state.choice(len(all_candidates), size=n_iter, replace=False)
    return [all_candidates[int(index)] for index in selected_indices]


def build_tuning_experiment_name(
    *,
    feature_set_name: str,
    selection_mode: str,
    n_iter: int,
    search_seed: int,
    eval_seeds: list[int],
) -> str:
    return normalize_experiment_name(
        f"{feature_set_name}__figs__{selection_mode}__n{n_iter}__search{search_seed}__eval{format_seed_suffix(eval_seeds)}"
    )


def build_training_experiment_name(*, feature_set_name: str, final_seed: int) -> str:
    return normalize_experiment_name(f"{feature_set_name}__figs__train__seed{final_seed}")


def build_multi_scheme_experiment_name(
    *,
    selection_mode: str,
    n_iter: int,
    search_seed: int,
    eval_seeds: list[int],
) -> str:
    return normalize_experiment_name(
        f"default_feature_schemes__figs__{selection_mode}__n{n_iter}__search{search_seed}__eval{format_seed_suffix(eval_seeds)}"
    )


def build_figs_model(params: dict[str, object], *, random_state: int | None):
    require_figs()
    return FIGSClassifier(
        **params,
        random_state=random_state,
    )


def get_positive_class_scores(model, x_matrix) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability_matrix = np.asarray(model.predict_proba(x_matrix), dtype=float)
        if probability_matrix.ndim == 1:
            return probability_matrix

        class_labels = list(getattr(model, "classes_", []))
        if 1 in class_labels:
            positive_class_index = class_labels.index(1)
        else:
            positive_class_index = probability_matrix.shape[1] - 1
        return probability_matrix[:, positive_class_index]

    predictions = np.asarray(model.predict(x_matrix), dtype=float)
    return predictions


def evaluate_candidate(
    params,
    x_train,
    y_train,
    x_valid,
    y_valid,
    seeds: list[int],
    rf_jobs: int,
) -> dict[str, float]:
    del rf_jobs  # FIGS does not expose n_jobs; kept in the function signature for CLI compatibility.

    y_train_array = np.asarray(y_train, dtype=int)
    y_valid_array = np.asarray(y_valid, dtype=int)
    valid_macro_f1_scores = []
    valid_roc_auc_scores = []
    train_macro_f1_scores = []
    train_roc_auc_scores = []

    for seed in seeds:
        model = build_figs_model(params, random_state=seed)
        model.fit(x_train, y_train_array)

        train_scores = get_positive_class_scores(model, x_train)
        valid_scores = get_positive_class_scores(model, x_valid)
        train_predictions = model.predict(x_train)
        valid_predictions = model.predict(x_valid)

        train_metrics = compute_binary_classification_metrics(y_train_array, train_predictions, train_scores)
        valid_metrics = compute_binary_classification_metrics(y_valid_array, valid_predictions, valid_scores)

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
    del rf_jobs  # FIGS does not expose n_jobs; kept in the function signature for CLI compatibility.

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
            model = build_figs_model(params, random_state=seed)
            model.fit(x_fold_train, y_fold_train)

            train_scores = get_positive_class_scores(model, x_fold_train)
            valid_scores = get_positive_class_scores(model, x_fold_valid)
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
        "max_rules",
        "max_trees",
        "min_impurity_decrease",
        "max_features",
        "max_depth",
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
    selection_mode: str = "train_cv_5fold",
    rf_jobs: int = 1,
    scale_features: bool = False,
    output_dir: str | Path = "train/tree/results/default_experiment",
) -> dict[str, object]:
    require_figs()

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

    candidate_params = sample_parameter_configs(n_iter=n_iter, search_seed=search_seed)
    selection_mean_key = f"mean_valid_{selection_metric}"
    selection_std_key = f"std_valid_{selection_metric}"
    results = []
    best_record = None

    for candidate_index, params in enumerate(candidate_params, start=1):
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
    results_csv = task_output_dir / "figs_search_results.csv"
    results_df.to_csv(results_csv, index=False)

    best_params = extract_best_params(best_record)
    final_valid_report = None
    final_eval_seed = None
    if selection_mode == "train_cv_5fold":
        final_eval_seed = eval_seeds[0]
        final_model = build_figs_model(best_params, random_state=final_eval_seed)
        final_model.fit(x_train, np.asarray(y_train, dtype=int))

        train_scores = get_positive_class_scores(final_model, x_train)
        valid_scores = get_positive_class_scores(final_model, x_valid)
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
        "model_family": "figs",
        "model_class": "FIGSClassifier",
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
    del rf_jobs  # FIGS does not expose n_jobs; kept in the function signature for CLI compatibility.
    require_figs()

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

    model = build_figs_model(best_params, random_state=final_seed)
    y_train_array = np.asarray(y_train, dtype=int)
    y_valid_array = np.asarray(y_valid, dtype=int)
    model.fit(x_train, y_train_array)

    train_scores = get_positive_class_scores(model, x_train)
    valid_scores = get_positive_class_scores(model, x_valid)
    train_predictions = model.predict(x_train)
    valid_predictions = model.predict(x_valid)

    final_train_metrics = compute_binary_classification_metrics(y_train_array, train_predictions, train_scores)
    final_valid_metrics = compute_binary_classification_metrics(y_valid_array, valid_predictions, valid_scores)

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
        "model_family": "figs",
        "model_class": "FIGSClassifier",
        "model_params": best_params,
        "figs_params": best_params,
        "model": model,
        "metrics": {
            "train": final_train_metrics,
            "valid": final_valid_metrics,
        },
    }
    model_bundle_path = task_output_dir / "model_bundle.pkl"
    with model_bundle_path.open("wb") as handle:
        import pickle

        pickle.dump(model_bundle, handle)

    summary = {
        "task": task,
        "experiment_name": experiment_dir.name,
        "experiment_dir": str(experiment_dir.resolve()),
        "dataset_root": str(Path(dataset_root).resolve()),
        "train_split": train_split_name,
        "valid_split": valid_split_name,
        "final_seed": final_seed,
        "model_family": "figs",
        "model_class": "FIGSClassifier",
        "feature_set_name": feature_bundle["feature_set_name"],
        "feature_config_paths": feature_bundle["config_paths"],
        "loaded_feature_configs": feature_bundle["loaded_configs"],
        "feature_source": feature_bundle["feature_source"].describe(),
        "num_surviving_features": len(surviving_columns),
        "surviving_feature_names": surviving_columns,
        "model_params": best_params,
        "figs_params": best_params,
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
