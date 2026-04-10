#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


THIS_DIR = Path(__file__).resolve().parent
TREE_DIR = THIS_DIR.parent
if str(TREE_DIR) not in sys.path:
    sys.path.insert(0, str(TREE_DIR))

from feature_semantics import classify_feature_nlp_readiness, describe_feature_name
from task_semantics import load_task_label_semantics


DEFAULT_REASONING_ROOT = TREE_DIR / "tree_reasoning_processes"
DEFAULT_RESULTS_ROOT = TREE_DIR / "results"
DEFAULT_OUTPUT_ROOT = THIS_DIR / "outputs"
DEFAULT_TEMPLATE_DIR = THIS_DIR / "prompt_templates"

THRESHOLD_RESEARCH_RDKIT_RAW_NAMES = {
    "TPSA",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "MolLogP",
    "NumHDonors",
    "NumHAcceptors",
    "FractionCSP3",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "LabuteASA",
}

QUALITATIVE_REWRITE_RDKIT_RAW_NAMES = {
    "qed",
    "MinPartialCharge",
    "MaxPartialCharge",
    "MinAbsPartialCharge",
    "MaxAbsPartialCharge",
}

QUALITATIVE_REWRITE_PKA_RAW_NAMES = {
    "has_basic_site",
    "has_acidic_site",
    "is_amphoteric",
}



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect task-specific non-functional-group features from reasoning JSON files and "
            "prepare DeepResearch inputs for threshold playbook generation."
        )
    )
    parser.add_argument("--experiment-name", required=True, help="Experiment name under tree_reasoning_processes")
    parser.add_argument("--task", required=True, help="Task name under the experiment directory")
    parser.add_argument("--feature-set", required=True, help="Feature-set directory name under the task directory")
    parser.add_argument(
        "--reasoning-root",
        default=str(DEFAULT_REASONING_ROOT),
        help="Root directory containing tree_reasoning_processes",
    )
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Root directory containing task result artifacts such as best_params.json and train_summary.json",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory where the organized DeepResearch inputs will be written",
    )
    parser.add_argument(
        "--include-splits",
        default="valid",
        help="Comma-separated split prefixes to keep (for example: valid,train). Use 'all' to keep everything.",
    )
    parser.add_argument(
        "--include-source-families",
        default="rdkit,pka",
        help="Comma-separated source families to include in threshold research",
    )
    parser.add_argument(
        "--exclude-source-families",
        default="fg_top_level",
        help="Comma-separated source families to exclude even if present elsewhere",
    )
    parser.add_argument(
        "--max-threshold-examples",
        type=int,
        default=8,
        help="Maximum number of threshold examples to keep per feature",
    )
    parser.add_argument(
        "--max-important-examples",
        type=int,
        default=5,
        help="Maximum number of important-feature examples to keep per feature",
    )
    parser.add_argument(
        "--deepresearch-template",
        default=str(DEFAULT_TEMPLATE_DIR / "deepresearch_threshold_playbook_prompt_template.md"),
        help="Path to the DeepResearch prompt template",
    )
    parser.add_argument(
        "--feature-universe-source",
        choices=("hybrid", "results_surviving", "reasoning_observed"),
        default="hybrid",
        help=(
            "How to determine the task feature universe. "
            "'results_surviving' uses surviving_feature_names from task result artifacts, "
            "'reasoning_observed' uses only features already seen in reasoning JSON, "
            "and 'hybrid' prefers surviving_feature_names while keeping observed reasoning statistics."
        ),
    )
    parser.add_argument(
        "--render-filled-prompt",
        action="store_true",
        help="Render a task-specific DeepResearch prompt using the template and extracted feature list",
    )
    return parser.parse_args()


def parse_csv_argument(value: str) -> set[str]:
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def keep_sample(sample_identifier: str, allowed_splits: set[str] | None) -> bool:
    if allowed_splits is None:
        return True
    return any(sample_identifier.startswith(f"{split_name}_") for split_name in allowed_splits)


def reasoning_json_paths(
    reasoning_root: Path,
    experiment_name: str,
    task: str,
    feature_set: str,
) -> list[Path]:
    task_dir = reasoning_root / experiment_name / task / feature_set
    if not task_dir.exists():
        raise FileNotFoundError(f"Reasoning directory not found: {task_dir}")
    return sorted(task_dir.glob("*.json"))


def result_artifact_paths(results_root: Path, experiment_name: str, task: str, feature_set: str) -> list[Path]:
    result_dir = results_root / experiment_name / task / feature_set
    candidate_paths = [
        result_dir / "best_params.json",
        result_dir / "train_summary.json",
    ]
    return [path for path in candidate_paths if path.exists()]


def make_output_dir(output_root: Path, experiment_name: str, task: str, feature_set: str) -> Path:
    output_dir = output_root / experiment_name / task / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_feature_record(feature_name: str) -> dict[str, object]:
    semantics = describe_feature_name(feature_name)
    return {
        "feature_name": feature_name,
        "raw_name": semantics["raw_name"],
        "display_name": semantics["display_name"],
        "description": semantics["description"],
        "source_family": semantics["source_family"],
        "nlp_readiness": classify_feature_nlp_readiness(feature_name),
        "important_feature_occurrences": 0,
        "reasoning_step_occurrences": 0,
        "sample_identifiers": set(),
        "important_examples": [],
        "threshold_examples": [],
        "threshold_signatures": set(),
        "operator_counts": Counter(),
        "direction_counts": Counter(),
        "feature_value_examples": [],
        "abs_shap_values": [],
    }


def classify_research_track(source_family: str, raw_name: str) -> tuple[str, str]:
    if source_family == "fg_top_level":
        return (
            "excluded_from_threshold_research",
            "functional-group indicator/count features are excluded from literature-threshold research by policy",
        )

    if source_family == "rdkit":
        if raw_name.startswith("fr_"):
            return (
                "qualitative_rewrite",
                "RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets",
            )
        if raw_name in THRESHOLD_RESEARCH_RDKIT_RAW_NAMES:
            return (
                "threshold_research",
                "classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics",
            )
        if raw_name in QUALITATIVE_REWRITE_RDKIT_RAW_NAMES:
            return (
                "qualitative_rewrite",
                "descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution",
            )
        return (
            "qualitative_rewrite",
            "RDKit feature kept for qualitative rewriting support because literature-threshold support is uncertain",
        )

    if source_family == "pka":
        if raw_name.startswith("warning_") or raw_name in QUALITATIVE_REWRITE_PKA_RAW_NAMES:
            return (
                "qualitative_rewrite",
                "pKa warning or boolean state feature is better used as qualitative support than as a threshold target",
            )
        return (
            "threshold_research",
            "pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research",
        )

    return (
        "qualitative_rewrite",
        "feature family is not in the main threshold-research set, so keep it only for qualitative rewriting support",
    )


def maybe_append_limited(target_list: list[dict[str, object]], payload: dict[str, object], limit: int) -> None:
    if len(target_list) < limit:
        target_list.append(payload)


def format_number(value: object) -> str:
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


def summarize_counter(counter: Counter[str]) -> list[dict[str, object]]:
    return [
        {"name": name, "count": count}
        for name, count in counter.most_common()
    ]


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def load_surviving_feature_universe(
    *,
    results_root: Path,
    experiment_name: str,
    task: str,
    feature_set: str,
) -> tuple[list[str], list[str]]:
    artifact_paths = result_artifact_paths(results_root, experiment_name, task, feature_set)
    if not artifact_paths:
        return [], []

    surviving_features: list[str] = []
    seen: set[str] = set()
    used_artifacts: list[str] = []
    for artifact_path in artifact_paths:
        payload = load_json(artifact_path)
        raw_names = payload.get("surviving_feature_names")
        if not isinstance(raw_names, list):
            continue
        used_artifacts.append(str(artifact_path))
        for feature_name in raw_names:
            feature_name = str(feature_name)
            if feature_name in seen:
                continue
            seen.add(feature_name)
            surviving_features.append(feature_name)
    return surviving_features, used_artifacts


def build_feature_list_markdown(feature_rows: list[dict[str, object]]) -> str:
    lines = []
    for row in feature_rows:
        lines.append(
            (
                f"- `{row['feature_name']}`: {row['display_name']} "
                f"(source family: {row['source_family']}; raw name: {row['raw_name']}; "
                f"description: {row['description']})"
            )
        )
    return "\n".join(lines)


def build_concise_feature_list_markdown(feature_rows: list[dict[str, object]]) -> str:
    seen_display_names: set[str] = set()
    lines = []
    for row in feature_rows:
        display_name = str(row["display_name"])
        if display_name in seen_display_names:
            continue
        seen_display_names.add(display_name)
        lines.append(f"- {display_name}")
    return "\n".join(lines)


def build_alias_map_markdown(feature_rows: list[dict[str, object]]) -> str:
    lines = []
    for row in feature_rows:
        lines.append(f"- `{row['feature_name']}`")
        lines.append(f"  canonical_name: {row['display_name']}")
        lines.append(f"  raw_name: {row['raw_name']}")
        lines.append(f"  source_family: {row['source_family']}")
        lines.append(f"  description: {row['description']}")
    return "\n".join(lines)


def render_template(template_text: str, replacements: dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def write_text_checked(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    written_text = path.read_text(encoding="utf-8")
    if written_text != text:
        raise RuntimeError(f"Failed to persist expected content to {path}")


def main() -> int:
    args = parse_args()

    reasoning_root = Path(args.reasoning_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    template_path = Path(args.deepresearch_template).expanduser().resolve()

    allowed_splits = parse_csv_argument(args.include_splits)
    if "all" in allowed_splits:
        allowed_splits_filter = None
    else:
        allowed_splits_filter = allowed_splits

    include_source_families = parse_csv_argument(args.include_source_families)
    exclude_source_families = parse_csv_argument(args.exclude_source_families)

    surviving_feature_universe, feature_universe_artifacts = load_surviving_feature_universe(
        results_root=results_root,
        experiment_name=args.experiment_name,
        task=args.task,
        feature_set=args.feature_set,
    )

    json_paths: list[Path] = []
    if args.feature_universe_source in {"hybrid", "reasoning_observed"}:
        json_paths = reasoning_json_paths(
            reasoning_root=reasoning_root,
            experiment_name=args.experiment_name,
            task=args.task,
            feature_set=args.feature_set,
        )

    feature_records: dict[str, dict[str, object]] = {}
    for feature_name in surviving_feature_universe:
        record = feature_records.setdefault(feature_name, build_feature_record(feature_name))
        record["in_surviving_feature_universe"] = True

    included_files = 0
    if args.feature_universe_source in {"hybrid", "reasoning_observed"}:
        for json_path in json_paths:
            payload = load_json(json_path)
            reasoning_schema = payload.get("reasoning_schema", {})
            sample_identifier = str(reasoning_schema.get("sample_identifier") or payload.get("sample_identifier") or json_path.stem)
            if not keep_sample(sample_identifier, allowed_splits_filter):
                continue

            included_files += 1

            for feature_row in reasoning_schema.get("important_feature_set", []):
                feature_name = str(feature_row["feature_name"])
                record = feature_records.setdefault(feature_name, build_feature_record(feature_name))
                source_family = str(record["source_family"])
                if source_family not in include_source_families or source_family in exclude_source_families:
                    continue
                record["important_feature_occurrences"] += 1
                record["sample_identifiers"].add(sample_identifier)
                record["abs_shap_values"].append(float(feature_row["abs_shap_value"]))
                maybe_append_limited(
                    record["important_examples"],
                    {
                        "sample_identifier": sample_identifier,
                        "rank": int(feature_row["rank"]),
                        "raw_value": feature_row["raw_value"],
                        "abs_shap_value": float(feature_row["abs_shap_value"]),
                    },
                    args.max_important_examples,
                )

            for tree_row in reasoning_schema.get("candidate_trees", []):
                for step_row in tree_row.get("reasoning_steps", []):
                    feature_name = str(step_row["feature_name"])
                    record = feature_records.setdefault(feature_name, build_feature_record(feature_name))
                    source_family = str(record["source_family"])
                    if source_family not in include_source_families or source_family in exclude_source_families:
                        continue

                    record["reasoning_step_occurrences"] += 1
                    record["sample_identifiers"].add(sample_identifier)
                    record["operator_counts"][str(step_row["comparison_operator"])] += 1
                    record["direction_counts"][str(step_row["next_node_majority_label_text"])] += 1

                    signature = (
                        format_number(step_row["threshold_raw_value"]),
                        str(step_row["comparison_operator"]),
                        str(step_row["next_node_majority_label_text"]),
                    )
                    if signature not in record["threshold_signatures"] and len(record["threshold_examples"]) < args.max_threshold_examples:
                        record["threshold_signatures"].add(signature)
                        record["threshold_examples"].append(
                            {
                                "sample_identifier": sample_identifier,
                                "tree_rank": int(tree_row["tree_rank"]),
                                "step_index": int(step_row["step_index"]),
                                "comparison_operator": str(step_row["comparison_operator"]),
                                "threshold_raw_value": step_row["threshold_raw_value"],
                                "sample_raw_value": step_row["sample_raw_value"],
                                "direction_label_text": str(step_row["next_node_majority_label_text"]),
                                "statement_for_sft": str(step_row["statement_for_sft"]),
                            }
                        )

                    if len(record["feature_value_examples"]) < args.max_threshold_examples:
                        record["feature_value_examples"].append(
                            {
                                "sample_identifier": sample_identifier,
                                "sample_raw_value": step_row["sample_raw_value"],
                                "threshold_raw_value": step_row["threshold_raw_value"],
                            }
                        )

    if args.feature_universe_source == "reasoning_observed" and included_files == 0:
        raise ValueError("No reasoning JSON files matched the requested split filter")
    if args.feature_universe_source in {"hybrid", "results_surviving"} and not surviving_feature_universe:
        raise ValueError(
            "No surviving_feature_names found in task result artifacts. "
            "Expected best_params.json or train_summary.json under the matching train/tree/results directory."
        )

    feature_rows = []
    for record in feature_records.values():
        source_family = str(record["source_family"])
        if source_family not in include_source_families or source_family in exclude_source_families:
            continue
        research_track, research_track_reason = classify_research_track(
            source_family=source_family,
            raw_name=str(record["raw_name"]),
        )
        feature_rows.append(
            {
                "feature_name": record["feature_name"],
                "raw_name": record["raw_name"],
                "display_name": record["display_name"],
                "description": record["description"],
                "source_family": source_family,
                "nlp_readiness": record["nlp_readiness"],
                "in_surviving_feature_universe": bool(record.get("in_surviving_feature_universe", False)),
                "observed_in_reasoning": (
                    int(record["important_feature_occurrences"]) > 0 or int(record["reasoning_step_occurrences"]) > 0
                ),
                "research_track": research_track,
                "research_track_reason": research_track_reason,
                "important_feature_occurrences": int(record["important_feature_occurrences"]),
                "reasoning_step_occurrences": int(record["reasoning_step_occurrences"]),
                "sample_count": len(record["sample_identifiers"]),
                "mean_abs_shap_value": mean(record["abs_shap_values"]),
                "operator_counts": summarize_counter(record["operator_counts"]),
                "direction_counts": summarize_counter(record["direction_counts"]),
                "important_examples": record["important_examples"],
                "threshold_examples": record["threshold_examples"],
                "feature_value_examples": record["feature_value_examples"],
            }
        )

    feature_rows.sort(
        key=lambda row: (
            row["in_surviving_feature_universe"],
            row["reasoning_step_occurrences"],
            row["important_feature_occurrences"],
            row["sample_count"],
            row["mean_abs_shap_value"] or -1.0,
            row["feature_name"],
        ),
        reverse=True,
    )

    if args.feature_universe_source == "reasoning_observed":
        playbook_candidate_rows = feature_rows
    else:
        playbook_candidate_rows = [
            row for row in feature_rows
            if row["in_surviving_feature_universe"]
        ]

    threshold_research_features = [
        row for row in playbook_candidate_rows
        if row["research_track"] == "threshold_research"
    ]
    qualitative_rewrite_features = [
        row for row in playbook_candidate_rows
        if row["research_track"] == "qualitative_rewrite"
    ]

    label_semantics = load_task_label_semantics(args.task)
    if label_semantics is None:
        class_a_text = "class 0"
        class_b_text = "class 1"
    else:
        class_a_text = label_semantics[0]["text"]
        class_b_text = label_semantics[1]["text"]

    output_dir = make_output_dir(
        output_root=output_root,
        experiment_name=args.experiment_name,
        task=args.task,
        feature_set=args.feature_set,
    )

    summary_payload = {
        "experiment_name": args.experiment_name,
        "task": args.task,
        "feature_set": args.feature_set,
        "feature_universe_source": args.feature_universe_source,
        "reasoning_root": str(reasoning_root),
        "results_root": str(results_root),
        "feature_universe_artifacts": feature_universe_artifacts,
        "included_splits": None if allowed_splits_filter is None else sorted(allowed_splits_filter),
        "included_source_families": sorted(include_source_families),
        "excluded_source_families": sorted(exclude_source_families),
        "functional_groups_excluded_from_threshold_research": "fg_top_level" in exclude_source_families,
        "num_reasoning_files_scanned": len(json_paths),
        "num_reasoning_files_included": included_files,
        "num_surviving_features_in_task_results": len(surviving_feature_universe),
        "task_semantics": {
            "class_a_text": class_a_text,
            "class_b_text": class_b_text,
        },
        "all_collected_features": feature_rows,
        "playbook_candidate_features": playbook_candidate_rows,
        "threshold_research_features": threshold_research_features,
        "qualitative_rewrite_features": qualitative_rewrite_features,
    }

    summary_json_path = output_dir / "task_threshold_research_brief.json"
    write_text_checked(summary_json_path, json.dumps(summary_payload, indent=2, ensure_ascii=False))

    markdown_lines = [
        f"# Threshold Research Brief: {args.task}",
        "",
        f"- Experiment: `{args.experiment_name}`",
        f"- Feature set: `{args.feature_set}`",
        f"- Feature universe source: `{args.feature_universe_source}`",
        f"- Included reasoning files: {included_files}",
        f"- Surviving features from task results: {len(surviving_feature_universe)}",
        f"- Playbook candidate features: {len(playbook_candidate_rows)}",
        f"- Included source families: {', '.join(sorted(include_source_families))}",
        f"- Excluded source families: {', '.join(sorted(exclude_source_families)) or 'none'}",
        f"- Class A: {class_a_text}",
        f"- Class B: {class_b_text}",
        "",
        "## Research Scope",
        "",
        "- Prefer the task's surviving feature universe from result artifacts as the stable source of candidate features.",
        "- Use reasoning JSON observations as priority signals, not as the sole feature-discovery source.",
        "- Focus literature threshold collection on the features below.",
        "- Do not search literature thresholds for functional-group indicators/counts from `fg_top_level`.",
        "- Use threshold examples from reasoning data only as model-side context, not as evidence.",
        "",
        "## Threshold-Research Features",
        "",
        build_feature_list_markdown(threshold_research_features) if threshold_research_features else "- No threshold-research features found.",
        "",
        "## Qualitative-Rewrite Features",
        "",
        build_feature_list_markdown(qualitative_rewrite_features) if qualitative_rewrite_features else "- No qualitative-rewrite features found.",
        "",
        "## Observed Threshold Examples From Reasoning Data",
        "",
    ]

    if feature_rows:
        for row in feature_rows:
            markdown_lines.extend(
                [
                    f"### `{row['feature_name']}`",
                    "",
                    f"- Display name: {row['display_name']}",
                    f"- Description: {row['description']}",
                    f"- Source family: {row['source_family']}",
                    f"- In surviving feature universe: {row['in_surviving_feature_universe']}",
                    f"- Observed in reasoning JSON: {row['observed_in_reasoning']}",
                    f"- Research track: {row['research_track']}",
                    f"- Track rationale: {row['research_track_reason']}",
                    f"- Important-feature occurrences: {row['important_feature_occurrences']}",
                    f"- Reasoning-step occurrences: {row['reasoning_step_occurrences']}",
                    f"- Sample count: {row['sample_count']}",
                    "",
                ]
            )
            if row["threshold_examples"]:
                markdown_lines.append("- Threshold examples:")
                for example in row["threshold_examples"]:
                    markdown_lines.append(
                        (
                            f"  - sample `{example['sample_identifier']}`, tree rank {example['tree_rank']}, "
                            f"step {example['step_index']}: value {format_number(example['sample_raw_value'])} "
                            f"{example['comparison_operator']} threshold {format_number(example['threshold_raw_value'])} "
                            f"-> {example['direction_label_text']}"
                        )
                    )
            else:
                markdown_lines.append("- Threshold examples: none captured in selected reasoning steps.")
            markdown_lines.append("")
    else:
        markdown_lines.append("- No eligible features found.")

    markdown_path = output_dir / "task_threshold_research_brief.md"
    write_text_checked(markdown_path, "\n".join(markdown_lines).rstrip() + "\n")

    if args.render_filled_prompt:
        template_text = template_path.read_text(encoding="utf-8")
        filled_prompt = render_template(
            template_text,
            {
                "TASK_NAME": args.task,
                "CLASS_A_TEXT": class_a_text,
                "CLASS_B_TEXT": class_b_text,
                "FEATURE_LIST_WITH_ALIASES": (
                    build_concise_feature_list_markdown(threshold_research_features)
                    if threshold_research_features
                    else "- No threshold-research features found."
                ),
                "FUNCTIONAL_GROUP_EXCLUSION_NOTE": (
                    "Do not spend literature search budget on `fg_top_level` functional-group indicator/count features; "
                    "we do not need literature thresholds for those."
                ),
            },
        )
        filled_prompt_path = output_dir / "deepresearch_threshold_playbook_prompt_filled.md"
        write_text_checked(filled_prompt_path, filled_prompt)
    else:
        filled_prompt_path = None

    print(
        json.dumps(
            {
                "summary_json": str(summary_json_path),
                "summary_markdown": str(markdown_path),
                "filled_prompt": None if filled_prompt_path is None else str(filled_prompt_path),
                "num_all_features": len(feature_rows),
                "num_playbook_candidate_features": len(playbook_candidate_rows),
                "num_threshold_research_features": len(threshold_research_features),
                "num_qualitative_rewrite_features": len(qualitative_rewrite_features),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
