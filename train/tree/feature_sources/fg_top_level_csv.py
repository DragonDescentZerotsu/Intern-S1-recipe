from __future__ import annotations

from pathlib import Path

from .csv_smiles_lookup import build_csv_smiles_lookup_source
from .registry import register_feature_source


DEFAULT_FG_TOP_LEVEL_CSV = Path(
    "/data1/tianang/Projects/AccFG/FG_feature_extraction/extracted_FG_features/"
    "tdc_no_conflict_labels_salt_removed_unique_smiles_top_level_fg_vectors.csv"
)


def build_fg_top_level_csv_source(spec: dict[str, object]):
    source_spec = dict(spec)
    source_spec.pop("source_type", None)
    source_spec.setdefault("name", "fg_top_level")
    source_spec.setdefault("csv_path", str(DEFAULT_FG_TOP_LEVEL_CSV))
    source_spec.setdefault("smiles_column", "smiles")
    source_spec.setdefault("prefix", "fg_top_level__")
    return build_csv_smiles_lookup_source(source_spec)


register_feature_source("fg_top_level_csv", build_fg_top_level_csv_source)

