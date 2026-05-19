from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train.tree.feature_semantics import build_feature_semantics_map


DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "DataPrepare"
    / "mol_features_for_tree"
    / "rdkit_descriptors_and_pka"
    / "tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka.csv"
)

DEFAULT_OUTPUT_ROOT_LV1 = (
    PROJECT_DIR
    / "DataPrepare"
    / "mol_features_for_tree"
    / "rdkit_descriptors_and_pka_easy_to_NLP_Lv1"
)

DEFAULT_OUTPUT_ROOT_LV2 = (
    PROJECT_DIR
    / "DataPrepare"
    / "mol_features_for_tree"
    / "rdkit_descriptors_and_pka_easy_to_NLP_Lv2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter existing RDKit+pKa feature CSV into NLP-friendlier Lv1/Lv2 variants."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output-root-lv1", type=Path, default=DEFAULT_OUTPUT_ROOT_LV1)
    parser.add_argument("--output-root-lv2", type=Path, default=DEFAULT_OUTPUT_ROOT_LV2)
    parser.add_argument(
        "--output-stem",
        default="tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka",
        help="Base stem reused in output filenames.",
    )
    return parser.parse_args()


def choose_columns_by_level(feature_columns: list[str]) -> tuple[list[str], list[str], dict[str, dict[str, str]]]:
    semantics_map = build_feature_semantics_map([f"rdkit_pka__{column}" for column in feature_columns])

    lv1_columns = []
    lv2_columns = []
    for raw_column in feature_columns:
        readiness = semantics_map[f"rdkit_pka__{raw_column}"]["nlp_readiness"]
        if readiness == "natural_language_ready":
            lv1_columns.append(raw_column)
            lv2_columns.append(raw_column)
            continue
        if readiness == "needs_translation":
            lv2_columns.append(raw_column)

    return lv1_columns, lv2_columns, semantics_map


def write_level_outputs(
    *,
    level_name: str,
    input_csv: Path,
    output_root: Path,
    output_stem: str,
    source_frame: pd.DataFrame,
    smiles_column: str,
    selected_columns: list[str],
    prefixed_semantics_map: dict[str, dict[str, str]],
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / f"{output_stem}_{level_name}.csv"
    metadata_path = output_root / f"{output_stem}_{level_name}.metadata.json"

    output_frame = source_frame[[smiles_column, *selected_columns]].copy()
    output_frame.to_csv(csv_path, index=False)

    readiness_counts: dict[str, int] = {}
    feature_semantics = {}
    for raw_column in selected_columns:
        prefixed_name = f"rdkit_pka__{raw_column}"
        semantics = dict(prefixed_semantics_map[prefixed_name])
        readiness = semantics["nlp_readiness"]
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
        feature_semantics[raw_column] = semantics

    metadata = {
        "level_name": level_name,
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(csv_path.resolve()),
        "smiles_column": smiles_column,
        "num_rows": int(len(output_frame)),
        "feature_column_count": int(len(selected_columns)),
        "selected_feature_columns": selected_columns,
        "readiness_counts": readiness_counts,
        "feature_semantics": feature_semantics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    if args.smiles_column not in frame.columns:
        raise KeyError(f"Missing smiles column {args.smiles_column!r} in {args.input_csv}")

    feature_columns = [column for column in frame.columns if column != args.smiles_column]
    lv1_columns, lv2_columns, semantics_map = choose_columns_by_level(feature_columns)

    metadata_lv1 = write_level_outputs(
        level_name="easy_to_NLP_Lv1",
        input_csv=args.input_csv,
        output_root=args.output_root_lv1,
        output_stem=args.output_stem,
        source_frame=frame,
        smiles_column=args.smiles_column,
        selected_columns=lv1_columns,
        prefixed_semantics_map=semantics_map,
    )
    metadata_lv2 = write_level_outputs(
        level_name="easy_to_NLP_Lv2",
        input_csv=args.input_csv,
        output_root=args.output_root_lv2,
        output_stem=args.output_stem,
        source_frame=frame,
        smiles_column=args.smiles_column,
        selected_columns=lv2_columns,
        prefixed_semantics_map=semantics_map,
    )

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "input_feature_count": len(feature_columns),
        "lv1_feature_count": len(lv1_columns),
        "lv2_feature_count": len(lv2_columns),
        "lv1_output_csv": metadata_lv1["output_csv"],
        "lv2_output_csv": metadata_lv2["output_csv"],
        "dropped_for_lv1": len(feature_columns) - len(lv1_columns),
        "dropped_for_lv2": len(feature_columns) - len(lv2_columns),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
