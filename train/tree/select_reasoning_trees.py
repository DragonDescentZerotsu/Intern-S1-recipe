#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data import DEFAULT_DATA_ROOT, DEFAULT_LABEL_FIELD, DEFAULT_SMILES_FIELD, get_tdc_split_sample
from feature_semantics import build_feature_semantics_map, describe_feature_name
from rf_pipeline import load_model_bundle, select_reasoning_trees_with_model_bundle
from task_semantics import load_task_label_semantics


def format_number(value) -> str:
    if value is None:
        return "NA"
    numeric_value = float(value)
    if abs(numeric_value - round(numeric_value)) < 1e-9:
        return str(int(round(numeric_value)))
    if abs(numeric_value) >= 100:
        return f"{numeric_value:.2f}"
    if abs(numeric_value) >= 1:
        return f"{numeric_value:.4f}".rstrip("0").rstrip(".")
    return f"{numeric_value:.6f}".rstrip("0").rstrip(".")


def build_sample_tag(payload: dict[str, object]) -> str:
    if payload.get("input_mode") == "tdc_split":
        return f"{payload['split']}_sample_{payload['sample_index']}"
    return "manual_smiles_sample"


def infer_experiment_name(args: argparse.Namespace, bundle_path: Path) -> str:
    if args.experiment_name:
        return str(args.experiment_name)

    resolved_parts = bundle_path.resolve().parts
    for anchor in ("bundles", "results"):
        if anchor in resolved_parts:
            anchor_index = resolved_parts.index(anchor)
            if anchor_index + 1 < len(resolved_parts):
                return resolved_parts[anchor_index + 1]
    return "manual_bundle"


def resolve_output_path(args: argparse.Namespace, payload: dict[str, object], bundle_path: Path) -> Path:
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (Path.cwd() / output_path).resolve()
        else:
            output_path = output_path.resolve()
        return output_path

    experiment_name = infer_experiment_name(args, bundle_path)
    task_name = str(payload.get("task") or args.task or "unknown_task")
    feature_set_name = str(payload.get("feature_set_name") or "unknown_feature_set")
    sample_tag = build_sample_tag(payload)
    file_name = f"{sample_tag}__top{payload['shap_top_k']}__trees{len(payload['selected_trees'])}.json"

    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    else:
        output_root = output_root.resolve()
    return output_root / experiment_name / task_name / feature_set_name / file_name


def build_step_statement(step: dict[str, object], matched_feature: dict[str, object] | None) -> str:
    feature_name = step["feature_name"]
    operator = step["decision_operator"]
    branch_direction = "left" if step["decision_goes_left"] else "right"

    if step["sample_value_was_imputed"]:
        base_statement = (
            f"{feature_name} is missing in raw space, so the model uses the imputed model-input value "
            f"{format_number(step['sample_model_input_value'])}; this is {operator} the node threshold "
            f"{format_number(step['threshold_model_input'])}, so the path goes {branch_direction}."
        )
    else:
        base_statement = (
            f"{feature_name} has raw value {format_number(step['sample_raw_value'])}, which is {operator} "
            f"the raw threshold {format_number(step['threshold_raw_value'])}, so the path goes {branch_direction}."
        )

    if matched_feature is None:
        return base_statement

    return (
        f"{base_statement} This feature is also in the SHAP top set "
        f"(abs SHAP {format_number(matched_feature['abs_shap_value'])})."
    )


def describe_task_label(label_value, label_semantics: dict[int, dict[str, str]] | None) -> str:
    if label_semantics is None or label_value not in label_semantics:
        return f"class {label_value}"
    option_payload = label_semantics[label_value]
    return f"option ({option_payload['option']}): {option_payload['text']}"


def build_step_statement_with_semantics(
    step: dict[str, object],
    matched_feature: dict[str, object] | None,
    label_semantics: dict[int, dict[str, str]] | None,
    feature_display_name: str,
) -> str:
    branch_majority_class = step.get("next_node_majority_class")
    branch_majority_text = describe_task_label(branch_majority_class, label_semantics)

    if step["sample_value_was_imputed"]:
        base_statement = (
            f"{feature_display_name} is missing in raw space, so the model uses the imputed model-input value "
            f"{format_number(step['sample_model_input_value'])}; this satisfies the split and moves the path "
            f"toward {branch_majority_text}."
        )
    else:
        base_statement = (
            f"{feature_display_name} has raw value {format_number(step['sample_raw_value'])}, which is "
            f"{step['decision_operator']} the raw threshold {format_number(step['threshold_raw_value'])}; "
            f"this steers the path toward {branch_majority_text}."
        )

    if matched_feature is None:
        return base_statement

    return (
        f"{base_statement} This feature is also in the SHAP top set "
        f"(abs SHAP {format_number(matched_feature['abs_shap_value'])})."
    )


def build_path_level_reasoning_note(
    *,
    tree: dict[str, object],
    target_class_text: str,
    matched_important_features: list[dict[str, object]],
    reasoning_steps: list[dict[str, object]],
) -> str:
    transition_words = ["First", "Next", "Then", "After that", "Finally"]

    if matched_important_features:
        top_feature_text = ", ".join(
            feature["feature_display_name"]
            for feature in matched_important_features[:3]
        )
        intro_clause = (
            f"This tree's reasoning repeatedly relies on important features such as {top_feature_text}."
        )
    else:
        intro_clause = "This tree's reasoning is driven by the ordered feature checks below."

    if reasoning_steps:
        step_clauses = []
        for step_index, step in enumerate(reasoning_steps):
            transition = (
                transition_words[step_index]
                if step_index < len(transition_words)
                else f"Step {step_index + 1}"
            )
            step_statement = str(step["statement_for_sft"]).rstrip()
            if step_statement.endswith("."):
                step_statement = step_statement[:-1]
            step_clauses.append(f"{transition}, {step_statement}.")
        detail_clause = " ".join(step_clauses)
    else:
        detail_clause = "The tree reaches its leaf without any non-leaf feature checks."

    return (
        f"{intro_clause} {detail_clause} Under this sequence of conditions, the tree supports {target_class_text}."
    )


def build_reasoning_schema(payload: dict[str, object]) -> dict[str, object]:
    task_name = payload.get("task")
    label_semantics = None if task_name is None else load_task_label_semantics(str(task_name))
    feature_semantics = build_feature_semantics_map(
        [feature_row["feature_name"] for feature_row in payload["shap_top_features"]]
    )
    for tree in payload["selected_trees"]:
        for feature_name in tree["decision_path"]["decision_features"]:
            feature_semantics.setdefault(feature_name, describe_feature_name(feature_name))

    shap_rank_lookup = {
        feature_row["feature_name"]: rank
        for rank, feature_row in enumerate(payload["shap_top_features"], start=1)
    }
    shap_feature_lookup = {
        feature_row["feature_name"]: feature_row
        for feature_row in payload["shap_top_features"]
    }

    important_feature_set = []
    for rank, feature_row in enumerate(payload["shap_top_features"], start=1):
        important_feature_set.append(
            {
                "rank": rank,
                "feature_name": feature_row["feature_name"],
                "feature_display_name": feature_semantics[feature_row["feature_name"]]["display_name"],
                "feature_description": feature_semantics[feature_row["feature_name"]]["description"],
                "raw_value": feature_row["raw_value"],
                "model_input_value": feature_row["model_input_value"],
                "abs_shap_value": feature_row["abs_shap_value"],
                "statement_for_sft": (
                    f"Important feature {rank}: {feature_semantics[feature_row['feature_name']]['display_name']} "
                    f"({feature_semantics[feature_row['feature_name']]['description']}) has raw value "
                    f"{format_number(feature_row['raw_value'])} and absolute SHAP contribution "
                    f"{format_number(feature_row['abs_shap_value'])}. This evidence is relevant to "
                    f"{describe_task_label(payload.get('explained_class'), label_semantics)}."
                ),
            }
        )

    candidate_trees = []
    for tree_rank, tree in enumerate(payload["selected_trees"], start=1):
        matched_important_features = []
        for feature_row in tree["hit_features"]:
            matched_important_features.append(
                {
                    "feature_name": feature_row["feature_name"],
                    "feature_display_name": feature_semantics[feature_row["feature_name"]]["display_name"],
                    "feature_description": feature_semantics[feature_row["feature_name"]]["description"],
                    "shap_rank": shap_rank_lookup[feature_row["feature_name"]],
                    "raw_value": feature_row["raw_value"],
                    "model_input_value": feature_row["model_input_value"],
                    "abs_shap_value": feature_row["abs_shap_value"],
                }
            )

        reasoning_steps = []
        for step_index, step in enumerate(tree["decision_path"]["steps"], start=1):
            if step["is_leaf"]:
                continue
            matched_feature = shap_feature_lookup.get(step["feature_name"])
            semantics = feature_semantics[step["feature_name"]]
            reasoning_steps.append(
                {
                    "step_index": step_index,
                    "feature_name": step["feature_name"],
                    "feature_display_name": semantics["display_name"],
                    "feature_description": semantics["description"],
                    "comparison_operator": step["decision_operator"],
                    "branch_direction": "left" if step["decision_goes_left"] else "right",
                    "sample_raw_value": step.get("sample_raw_value"),
                    "sample_model_input_value": step["sample_model_input_value"],
                    "threshold_raw_value": step["threshold_raw_value"],
                    "threshold_model_input": step["threshold_model_input"],
                    "sample_value_was_imputed": step["sample_value_was_imputed"],
                    "is_shap_top_feature": matched_feature is not None,
                    "shap_rank": None if matched_feature is None else shap_rank_lookup[step["feature_name"]],
                    "abs_shap_value": None if matched_feature is None else matched_feature["abs_shap_value"],
                    "next_node_majority_class": step.get("next_node_majority_class"),
                    "next_node_majority_class_probability": step.get("next_node_majority_class_probability"),
                    "next_node_majority_label_text": describe_task_label(
                        step.get("next_node_majority_class"),
                        label_semantics,
                    ),
                    "statement_for_sft": build_step_statement_with_semantics(
                        step,
                        matched_feature,
                        label_semantics,
                        semantics["display_name"],
                    ),
                }
            )

        target_class = payload.get("explained_class", payload.get("forest_prediction"))
        target_class_text = describe_task_label(target_class, label_semantics)
        path_level_reasoning_note = build_path_level_reasoning_note(
            tree=tree,
            target_class_text=target_class_text,
            matched_important_features=matched_important_features,
            reasoning_steps=reasoning_steps,
        )
        candidate_trees.append(
            {
                "tree_rank": tree_rank,
                "tree_index": tree["tree_index"],
                "selection_scores": {
                    "hit_count": tree["hit_count"],
                    "hit_abs_shap_sum": tree["hit_abs_shap_sum"],
                    "leaf_target_class_probability": tree["leaf_target_class_probability"],
                    "path_length": tree["path_length"],
                },
                "selection_statement_for_sft": (
                    f"Tree {tree['tree_index']} is selected at rank {tree_rank} because it matches "
                    f"{tree['hit_count']} SHAP-important features, accumulates "
                    f"{format_number(tree['hit_abs_shap_sum'])} absolute SHAP mass on those matches, "
                    f"and ends at a leaf that strongly supports {target_class_text} with probability "
                    f"{format_number(tree['leaf_target_class_probability'])}."
                ),
                "matched_important_features": matched_important_features,
                "reasoning_steps": reasoning_steps,
                "path_level_reasoning_note": path_level_reasoning_note,
                "conclusion": {
                    "predicted_class": tree["tree_prediction"],
                    "predicted_label_text": describe_task_label(tree["tree_prediction"], label_semantics),
                    "leaf_target_class_probability": tree["leaf_target_class_probability"],
                    "statement_for_sft": (
                        f"Following these conditions leads this tree to a leaf that predicts "
                        f"{describe_task_label(tree['tree_prediction'], label_semantics)} and assigns "
                        f"probability {format_number(tree['leaf_target_class_probability'])} to "
                        f"{target_class_text}."
                    ),
                },
            }
        )

    sample_identifier = build_sample_tag(payload)
    return {
        "schema_version": "tree_reasoning_v1",
        "sample_identifier": sample_identifier,
        "task": task_name,
        "feature_set_name": payload["feature_set_name"],
        "label_semantics": label_semantics,
        "feature_semantics": feature_semantics,
        "forest_requirement": {
            "require_forest_correct": payload["require_forest_correct"],
            "forest_prediction_correct": payload["forest_prediction_correct"],
            "skipped_reason": payload["skipped_reason"],
        },
        "sample_summary": {
            "smiles": payload["smiles"],
            "label": payload.get("label"),
            "label_text": describe_task_label(payload.get("label"), label_semantics),
            "forest_prediction": payload.get("forest_prediction"),
            "forest_prediction_text": describe_task_label(payload.get("forest_prediction"), label_semantics),
            "explained_class": payload.get("explained_class"),
            "explained_class_text": describe_task_label(payload.get("explained_class"), label_semantics),
            "explained_class_probability": payload.get("explained_class_probability"),
            "statement_for_sft": (
                f"For sample {sample_identifier}, the Random Forest predicts "
                f"{describe_task_label(payload.get('forest_prediction'), label_semantics)} with probability "
                f"{format_number(payload.get('explained_class_probability'))}."
            ),
        },
        "important_feature_set": important_feature_set,
        "candidate_trees": candidate_trees,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select sample-specific RandomForest trees that align with SHAP-important features."
    )
    parser.add_argument("--bundle", default=None, help="Path to model_bundle.pkl")
    parser.add_argument("--bundle-root", default=str(THIS_DIR / "bundles"), help="Root directory for exported bundles")
    parser.add_argument("--experiment-name", default=None, help="Experiment name under bundle-root")
    parser.add_argument("--task", default=None, help="Task name under bundle-root/<experiment>")
    parser.add_argument("--feature-set", default=None, help="Feature-set directory name. If omitted, infer when unique.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample index within a TDC split. Requires --split and a task.",
    )
    input_group.add_argument("--smiles", default=None, help="Analyze a single SMILES string")

    parser.add_argument("--split", default="valid", help="Dataset split used with --sample-index")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Root directory of TDC JSONL splits")
    parser.add_argument("--smiles-key", default=DEFAULT_SMILES_FIELD, help="SMILES field name in the JSONL files")
    parser.add_argument("--label-key", default=DEFAULT_LABEL_FIELD, help="Label field name in the JSONL files")
    parser.add_argument("--label", default=None, help="Optional true label to use with --smiles")
    parser.add_argument("--class-index", type=int, default=None, help="Optional class index to explain")
    parser.add_argument("--shap-top-k", type=int, default=30, help="Number of SHAP-important features to consider")
    parser.add_argument("--max-trees", type=int, default=5, help="Maximum number of trees to keep per sample")
    parser.add_argument(
        "--require-forest-correct",
        dest="require_forest_correct",
        action="store_true",
        help="Keep samples only when the full forest prediction matches the true label (default).",
    )
    parser.add_argument(
        "--allow-forest-incorrect",
        dest="require_forest_correct",
        action="store_false",
        help="Allow selecting trees even when the full forest prediction is wrong.",
    )
    parser.add_argument(
        "--output-root",
        default=str(THIS_DIR / "tree_reasoning_processes"),
        help="Root directory for saved reasoning-process payloads.",
    )
    parser.add_argument("--output-json", default=None, help="Optional path to write the selected-tree payload as JSON")
    parser.set_defaults(require_forest_correct=True)
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


def resolve_task(args: argparse.Namespace, bundle_path: Path) -> str:
    if args.task:
        return args.task
    bundle = load_model_bundle(bundle_path)
    task = bundle.get("task")
    if not task:
        raise ValueError("Could not infer task from model bundle; please pass --task explicitly")
    return str(task)


def parse_optional_label(label_text: str | None):
    if label_text is None:
        return None
    stripped = str(label_text).strip()
    if stripped in {"0", "1"}:
        return int(stripped)
    return stripped


def load_single_input(args: argparse.Namespace, bundle_path: Path) -> tuple[list[str], list[object] | None, dict[str, object]]:
    if args.smiles is not None:
        label = parse_optional_label(args.label)
        return [args.smiles], None if label is None else [label], {"input_mode": "smiles"}

    task = resolve_task(args, bundle_path)
    sample = get_tdc_split_sample(
        task=task,
        split=args.split,
        sample_index=args.sample_index,
        data_root=args.data_root,
        smiles_field=args.smiles_key,
        label_field=args.label_key,
    )
    return [str(sample["smiles"])], [sample["label"]], {
        "input_mode": "tdc_split",
        "task": sample["task"],
        "split": sample["split"],
        "sample_index": sample["sample_index"],
        "label": sample["label"],
    }


def main() -> int:
    args = parse_args()
    bundle_path = resolve_bundle_path(args)
    smiles_list, labels, metadata = load_single_input(args, bundle_path)

    results = select_reasoning_trees_with_model_bundle(
        smiles_list=smiles_list,
        labels=labels,
        bundle_path=bundle_path,
        shap_top_k=args.shap_top_k,
        max_trees=args.max_trees,
        class_index=args.class_index,
        require_forest_correct=args.require_forest_correct,
    )
    payload = results[0]
    payload["bundle_path"] = str(bundle_path)
    payload.update(metadata)
    payload["reasoning_schema"] = build_reasoning_schema(payload)

    output_path = resolve_output_path(args, payload, bundle_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"saved_to": str(output_path), "selected_trees": len(payload["selected_trees"])}, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
