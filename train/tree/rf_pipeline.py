from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
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
    feature_payload = load_feature_frames_with_model_bundle(
        smiles_list=smiles_list,
        model_bundle=model_bundle,
        bundle_path=bundle_path,
    )
    model_bundle = feature_payload["model_bundle"]
    transformed_frame = feature_payload["transformed_feature_frame"]

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


def load_feature_frames_with_model_bundle(
    *,
    smiles_list: list[str],
    model_bundle: dict[str, object] | None = None,
    bundle_path: str | Path | None = None,
) -> dict[str, object]:
    if model_bundle is None:
        if bundle_path is None:
            raise ValueError("Either model_bundle or bundle_path must be provided")
        model_bundle = load_model_bundle(bundle_path)

    feature_source = build_feature_source_for_model_bundle(model_bundle)
    raw_feature_frame = feature_source.load(smiles_list)
    aligned_feature_frame, transformed_feature_frame = transform_feature_frame(
        raw_feature_frame,
        model_bundle["preprocessor"],
    )
    return {
        "model_bundle": model_bundle,
        "raw_feature_frame": raw_feature_frame,
        "aligned_feature_frame": aligned_feature_frame,
        "transformed_feature_frame": transformed_feature_frame,
    }


def _load_shap_module():
    matplotlib_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "The shap package is required for TreeSHAP explanations. "
            "Please install shap in the active environment."
        ) from exc
    return shap


def _coerce_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coerce_json_scalar(value):
    value = _coerce_python_scalar(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _select_output_matrix(values: np.ndarray, class_indices: list[int]) -> np.ndarray:
    num_rows = len(class_indices)
    if values.ndim == 1:
        if num_rows != 1:
            raise ValueError(f"Expected one class index for 1D SHAP values, got {num_rows}.")
        return values.reshape(1, -1)
    if values.ndim == 2:
        if values.shape[0] == num_rows:
            return values
        if num_rows == 1:
            return values[[0], :]
        raise ValueError(
            f"2D SHAP values row count {values.shape[0]} does not match class index count {num_rows}."
        )
    if values.ndim == 3:
        if values.shape[0] == num_rows:
            return np.stack(
                [values[row_index, :, class_indices[row_index]] for row_index in range(num_rows)],
                axis=0,
            )
        if values.shape[1] == num_rows:
            return np.stack(
                [values[class_indices[row_index], row_index, :] for row_index in range(num_rows)],
                axis=0,
            )
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def _select_base_values(base_values: np.ndarray, class_indices: list[int]) -> np.ndarray:
    num_rows = len(class_indices)
    if base_values.ndim == 0:
        return np.full(num_rows, float(base_values))
    if base_values.ndim == 1:
        if len(base_values) == 1:
            return np.full(num_rows, float(base_values[0]))
        if num_rows == 1:
            return np.asarray([float(base_values[class_indices[0]])])
        if len(base_values) == num_rows:
            return np.asarray(base_values, dtype=float)
        raise ValueError(
            f"Unsupported 1D SHAP base value length {len(base_values)} for {num_rows} rows."
        )
    if base_values.ndim == 2:
        if base_values.shape[0] == num_rows:
            return np.asarray(
                [float(base_values[row_index, class_indices[row_index]]) for row_index in range(num_rows)],
                dtype=float,
            )
        if base_values.shape[1] == num_rows:
            return np.asarray(
                [float(base_values[class_indices[row_index], row_index]) for row_index in range(num_rows)],
                dtype=float,
            )
    raise ValueError(f"Unsupported SHAP base value shape: {base_values.shape}")


def _extract_model_outputs(
    model,
    transformed_array: np.ndarray,
) -> tuple[np.ndarray | None, list[object] | None, list[list[float]] | None, list[object] | None]:
    predicted_raw = None
    predicted_classes = None
    predicted_probabilities = None
    model_classes = None

    if hasattr(model, "predict"):
        predicted_raw = np.asarray(model.predict(transformed_array))
        predicted_classes = [_coerce_python_scalar(value) for value in predicted_raw.tolist()]
    if hasattr(model, "predict_proba"):
        probability_rows = np.asarray(model.predict_proba(transformed_array))
        predicted_probabilities = [
            [float(probability) for probability in probability_row]
            for probability_row in probability_rows.tolist()
        ]
    if hasattr(model, "classes_"):
        model_classes = [_coerce_python_scalar(value) for value in model.classes_.tolist()]

    return predicted_raw, predicted_classes, predicted_probabilities, model_classes


def _resolve_class_indices(
    *,
    class_index: int | None,
    predicted_classes: list[object] | None,
    model_classes: list[object] | None,
    num_rows: int,
) -> list[int]:
    if class_index is not None:
        return [class_index] * num_rows

    if predicted_classes is not None and model_classes is not None:
        class_indices = []
        for predicted_class in predicted_classes:
            if predicted_class in model_classes:
                class_indices.append(model_classes.index(predicted_class))
            else:
                class_indices.append(len(model_classes) - 1)
        return class_indices

    return [1] * num_rows


def explain_with_model_bundle(
    *,
    smiles_list: list[str],
    model_bundle: dict[str, object] | None = None,
    bundle_path: str | Path | None = None,
    class_index: int | None = 1,
    top_k: int | None = 20,
) -> list[dict[str, object]]:
    if not smiles_list:
        return []
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")

    shap = _load_shap_module()
    feature_payload = load_feature_frames_with_model_bundle(
        smiles_list=smiles_list,
        model_bundle=model_bundle,
        bundle_path=bundle_path,
    )
    model_bundle = feature_payload["model_bundle"]
    aligned_feature_frame = feature_payload["aligned_feature_frame"]
    transformed_feature_frame = feature_payload["transformed_feature_frame"]

    transformed_array = transformed_feature_frame.to_numpy()
    model = model_bundle["model"]
    predicted_raw, predicted_classes, predicted_probabilities, model_classes = _extract_model_outputs(
        model,
        transformed_array,
    )
    resolved_class_indices = _resolve_class_indices(
        class_index=class_index,
        predicted_classes=predicted_classes,
        model_classes=model_classes,
        num_rows=len(smiles_list),
    )

    explainer = shap.TreeExplainer(model)
    explanation = explainer(transformed_feature_frame)
    shap_matrix = _select_output_matrix(np.asarray(explanation.values), resolved_class_indices)
    base_values = _select_base_values(np.asarray(explanation.base_values), resolved_class_indices)

    feature_names = transformed_feature_frame.columns.tolist()
    results = []
    for row_index, smiles in enumerate(smiles_list):
        aligned_row = aligned_feature_frame.iloc[row_index]
        transformed_row = transformed_feature_frame.iloc[row_index]

        feature_rows = []
        for feature_name, shap_value in zip(feature_names, shap_matrix[row_index].tolist()):
            raw_value = aligned_row[feature_name]
            feature_rows.append(
                {
                    "feature_name": feature_name,
                    "raw_value": None if pd.isna(raw_value) else float(raw_value),
                    "model_input_value": float(transformed_row[feature_name]),
                    "shap_value": float(shap_value),
                    "abs_shap_value": abs(float(shap_value)),
                }
            )
        feature_rows.sort(key=lambda row: row["abs_shap_value"], reverse=True)
        if top_k is not None:
            feature_rows = feature_rows[:top_k]

        result = {
            "smiles": str(smiles),
            "feature_set_name": model_bundle["feature_set_name"],
            "class_index": int(resolved_class_indices[row_index]),
            "base_value": float(base_values[row_index]),
            "explained_output_value": float(base_values[row_index] + shap_matrix[row_index].sum()),
            "features": feature_rows,
        }
        if predicted_classes is not None:
            result["predicted_class"] = _coerce_json_scalar(predicted_classes[row_index])
        if predicted_probabilities is not None:
            result["predicted_probabilities"] = predicted_probabilities[row_index]
            explained_probability = predicted_probabilities[row_index][resolved_class_indices[row_index]]
            result["explained_class_probability"] = float(explained_probability)
        if model_classes is not None:
            result["model_classes"] = [_coerce_json_scalar(value) for value in model_classes]
            result["explained_class"] = _coerce_json_scalar(model_classes[resolved_class_indices[row_index]])
        elif predicted_raw is not None:
            result["predicted_value"] = float(predicted_raw[row_index])

        results.append(result)

    return results


def _get_feature_index_lookup(preprocessor: dict[str, object]) -> dict[str, int]:
    surviving_columns = list(preprocessor["surviving_columns"])
    return {feature_name: index for index, feature_name in enumerate(surviving_columns)}


def inverse_transform_feature_value(
    *,
    feature_name: str,
    transformed_value: float,
    preprocessor: dict[str, object],
) -> float:
    index_lookup = _get_feature_index_lookup(preprocessor)
    if feature_name not in index_lookup:
        raise KeyError(f"Feature {feature_name!r} is not present in the preprocessor surviving columns")

    feature_index = index_lookup[feature_name]
    scaler = preprocessor.get("scaler")
    value = float(transformed_value)
    if scaler is not None:
        value = value * float(scaler.scale_[feature_index]) + float(scaler.mean_[feature_index])
    return value


def _extract_single_tree_outputs(
    estimator,
    transformed_row: np.ndarray,
    *,
    target_class_index: int,
) -> dict[str, object]:
    predicted_class = _coerce_python_scalar(estimator.predict(transformed_row)[0])
    probability_row = estimator.predict_proba(transformed_row)[0]
    leaf_node_id = int(estimator.apply(transformed_row)[0])
    classes = [_coerce_python_scalar(value) for value in estimator.classes_.tolist()]
    class_probability_map = {
        _coerce_json_scalar(class_label): float(probability_row[class_index])
        for class_index, class_label in enumerate(classes)
    }
    return {
        "predicted_class": predicted_class,
        "predicted_probabilities": [float(probability) for probability in probability_row.tolist()],
        "leaf_node_id": leaf_node_id,
        "leaf_target_class_probability": float(probability_row[target_class_index]),
        "classes": classes,
        "class_probability_map": class_probability_map,
    }


def _build_tree_path_for_sample(
    *,
    estimator,
    aligned_row: pd.Series,
    transformed_row: pd.Series,
    preprocessor: dict[str, object],
) -> dict[str, object]:
    tree = estimator.tree_
    estimator_classes = [_coerce_python_scalar(value) for value in estimator.classes_.tolist()]
    transformed_array = transformed_row.to_numpy(dtype=float).reshape(1, -1)
    node_indicator = estimator.decision_path(transformed_array)
    node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]].tolist()

    steps = []
    for depth, node_id in enumerate(node_ids):
        feature_index = int(tree.feature[node_id])
        if feature_index < 0:
            steps.append(
                {
                    "depth": depth,
                    "node_id": int(node_id),
                    "is_leaf": True,
                    "leaf_node_id": int(node_id),
                    "n_node_samples": int(tree.n_node_samples[node_id]),
                }
            )
            continue

        feature_name = str(transformed_row.index[feature_index])
        threshold_model_input = float(tree.threshold[node_id])
        threshold_raw_value = inverse_transform_feature_value(
            feature_name=feature_name,
            transformed_value=threshold_model_input,
            preprocessor=preprocessor,
        )

        sample_model_input_value = float(transformed_row.iloc[feature_index])
        sample_raw_cell = aligned_row.iloc[feature_index]
        sample_raw_value = None if pd.isna(sample_raw_cell) else float(sample_raw_cell)
        sample_value_was_imputed = sample_raw_value is None
        decision_goes_left = sample_model_input_value <= threshold_model_input
        operator = "<=" if decision_goes_left else ">"
        next_node_id = int(tree.children_left[node_id] if decision_goes_left else tree.children_right[node_id])
        child_value_vector = np.asarray(tree.value[next_node_id][0], dtype=float)
        child_total = float(child_value_vector.sum())
        if child_total > 0:
            child_probabilities = [float(value / child_total) for value in child_value_vector.tolist()]
        else:
            child_probabilities = [0.0 for _ in child_value_vector.tolist()]
        branch_majority_class_index = int(np.argmax(child_value_vector)) if len(child_value_vector) else 0
        branch_majority_class = estimator_classes[branch_majority_class_index]
        branch_class_probability_map = {
            _coerce_json_scalar(class_label): float(child_probabilities[class_index])
            for class_index, class_label in enumerate(estimator_classes)
        }

        steps.append(
            {
                "depth": depth,
                "node_id": int(node_id),
                "is_leaf": False,
                "feature_name": feature_name,
                "feature_index": feature_index,
                "threshold_model_input": threshold_model_input,
                "threshold_raw_value": float(threshold_raw_value),
                "sample_model_input_value": sample_model_input_value,
                "sample_raw_value": sample_raw_value,
                "sample_value_was_imputed": sample_value_was_imputed,
                "decision_operator": operator,
                "decision_goes_left": decision_goes_left,
                "next_node_id": next_node_id,
                "n_node_samples": int(tree.n_node_samples[node_id]),
                "next_node_majority_class": _coerce_json_scalar(branch_majority_class),
                "next_node_majority_class_probability": float(child_probabilities[branch_majority_class_index]),
                "next_node_class_probability_map": branch_class_probability_map,
            }
        )

    decision_features = [
        step["feature_name"]
        for step in steps
        if not step["is_leaf"]
    ]
    return {
        "node_ids": [int(node_id) for node_id in node_ids],
        "steps": steps,
        "path_length": len(decision_features),
        "decision_features": decision_features,
    }


def select_reasoning_trees_with_model_bundle(
    *,
    smiles_list: list[str],
    labels: list[object] | None = None,
    model_bundle: dict[str, object] | None = None,
    bundle_path: str | Path | None = None,
    shap_top_k: int = 30,
    max_trees: int = 5,
    class_index: int | None = None,
    require_forest_correct: bool = True,
) -> list[dict[str, object]]:
    if not smiles_list:
        return []
    if labels is not None and len(labels) != len(smiles_list):
        raise ValueError("labels length must match smiles_list length")
    if shap_top_k <= 0:
        raise ValueError("shap_top_k must be positive")
    if max_trees <= 0:
        raise ValueError("max_trees must be positive")

    feature_payload = load_feature_frames_with_model_bundle(
        smiles_list=smiles_list,
        model_bundle=model_bundle,
        bundle_path=bundle_path,
    )
    model_bundle = feature_payload["model_bundle"]
    aligned_feature_frame = feature_payload["aligned_feature_frame"]
    transformed_feature_frame = feature_payload["transformed_feature_frame"]
    forest_model = model_bundle["model"]

    explanations = explain_with_model_bundle(
        smiles_list=smiles_list,
        model_bundle=model_bundle,
        class_index=class_index,
        top_k=shap_top_k,
    )

    results = []
    for sample_index, explanation in enumerate(explanations):
        aligned_row = aligned_feature_frame.iloc[sample_index]
        transformed_row = transformed_feature_frame.iloc[sample_index]
        transformed_array = transformed_row.to_numpy(dtype=float).reshape(1, -1)

        label = None if labels is None else _coerce_json_scalar(labels[sample_index])
        forest_predicted_class = explanation.get("predicted_class")
        forest_is_correct = None if label is None else (forest_predicted_class == label)

        feature_lookup = {
            feature_row["feature_name"]: feature_row
            for feature_row in explanation["features"]
        }
        shap_top_features = [feature_row["feature_name"] for feature_row in explanation["features"]]
        shap_abs_sum = float(sum(feature_row["abs_shap_value"] for feature_row in explanation["features"]))

        selected_tree_rows = []
        skipped_reason = None
        if require_forest_correct and forest_is_correct is False:
            skipped_reason = "forest_prediction_incorrect"
        else:
            target_class = explanation.get("explained_class", explanation.get("predicted_class"))
            model_classes = explanation.get("model_classes")
            if model_classes is not None and target_class in model_classes:
                target_class_index = model_classes.index(target_class)
            elif class_index is not None:
                target_class_index = int(class_index)
            else:
                target_class_index = 1

            for tree_index, estimator in enumerate(forest_model.estimators_):
                tree_output = _extract_single_tree_outputs(
                    estimator,
                    transformed_array,
                    target_class_index=target_class_index,
                )
                tree_is_correct = None if label is None else (tree_output["predicted_class"] == label)
                if label is not None and not tree_is_correct:
                    continue

                tree_path = _build_tree_path_for_sample(
                    estimator=estimator,
                    aligned_row=aligned_row,
                    transformed_row=transformed_row,
                    preprocessor=model_bundle["preprocessor"],
                )
                hit_features = []
                seen_hit_features: set[str] = set()
                for feature_name in tree_path["decision_features"]:
                    if feature_name not in feature_lookup or feature_name in seen_hit_features:
                        continue
                    seen_hit_features.add(feature_name)
                    hit_features.append(feature_lookup[feature_name])

                hit_count = len(hit_features)
                hit_abs_shap_sum = float(sum(feature_row["abs_shap_value"] for feature_row in hit_features))

                selected_tree_rows.append(
                    {
                        "tree_index": int(tree_index),
                        "tree_prediction": _coerce_json_scalar(tree_output["predicted_class"]),
                        "tree_prediction_correct": tree_is_correct,
                        "leaf_node_id": int(tree_output["leaf_node_id"]),
                        "leaf_target_class_probability": float(tree_output["leaf_target_class_probability"]),
                        "tree_predicted_probabilities": tree_output["predicted_probabilities"],
                        "tree_class_probability_map": tree_output["class_probability_map"],
                        "path_length": int(tree_path["path_length"]),
                        "hit_count": int(hit_count),
                        "hit_abs_shap_sum": float(hit_abs_shap_sum),
                        "hit_feature_names": [feature_row["feature_name"] for feature_row in hit_features],
                        "hit_features": hit_features,
                        "decision_path": tree_path,
                    }
                )

            selected_tree_rows.sort(
                key=lambda row: (
                    row["hit_count"],
                    row["hit_abs_shap_sum"],
                    row["leaf_target_class_probability"],
                    -row["path_length"],
                    -row["tree_index"],
                ),
                reverse=True,
            )
            selected_tree_rows = selected_tree_rows[:max_trees]

        results.append(
            {
                "sample_index": int(sample_index),
                "smiles": explanation["smiles"],
                "label": label,
                "forest_prediction": forest_predicted_class,
                "forest_prediction_correct": forest_is_correct,
                "require_forest_correct": require_forest_correct,
                "skipped_reason": skipped_reason,
                "feature_set_name": explanation["feature_set_name"],
                "explained_class": explanation.get("explained_class"),
                "class_index": explanation["class_index"],
                "explained_class_probability": explanation.get("explained_class_probability"),
                "predicted_probabilities": explanation.get("predicted_probabilities"),
                "base_value": explanation["base_value"],
                "explained_output_value": explanation["explained_output_value"],
                "shap_top_k": int(shap_top_k),
                "shap_top_feature_names": shap_top_features,
                "shap_top_abs_sum": shap_abs_sum,
                "shap_top_features": explanation["features"],
                "selected_trees": selected_tree_rows,
            }
        )

    return results


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
