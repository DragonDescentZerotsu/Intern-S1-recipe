from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import FeatureSource
from .registry import register_feature_source


class CSVSmilesLookupSource(FeatureSource):
    def __init__(
        self,
        *,
        name: str,
        csv_path: str | Path,
        smiles_column: str = "smiles",
        feature_columns: list[str] | None = None,
        drop_columns: list[str] | None = None,
        prefix: str = "",
    ):
        super().__init__(name=name)
        self.csv_path = Path(csv_path)
        self.smiles_column = smiles_column
        self.feature_columns = feature_columns
        self.drop_columns = set(drop_columns or [])
        self.prefix = prefix
        self._table: pd.DataFrame | None = None

    def _load_table(self) -> pd.DataFrame:
        if self._table is not None:
            return self._table

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find feature CSV: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        if self.smiles_column not in frame.columns:
            raise KeyError(
                f"Feature CSV {self.csv_path} is missing smiles column {self.smiles_column!r}"
            )

        if frame[self.smiles_column].duplicated().any():
            duplicated_count = int(frame[self.smiles_column].duplicated().sum())
            raise ValueError(
                f"Feature CSV {self.csv_path} contains {duplicated_count} duplicated SMILES entries"
            )

        if self.feature_columns is None:
            feature_columns = [
                column
                for column in frame.columns
                if column != self.smiles_column and column not in self.drop_columns
            ]
        else:
            feature_columns = list(self.feature_columns)

        if not feature_columns:
            raise ValueError(f"No feature columns were selected from {self.csv_path}")

        table = frame.set_index(self.smiles_column)[feature_columns]
        table.index = table.index.astype(str)
        renamed_columns = {
            column: f"{self.prefix}{column}"
            for column in table.columns
        }
        self._table = table.rename(columns=renamed_columns)
        return self._table

    def load(self, smiles_list) -> pd.DataFrame:
        table = self._load_table()
        ordered_smiles = [str(smiles) for smiles in smiles_list]
        aligned = table.reindex(ordered_smiles)

        missing_mask = aligned.isna().all(axis=1)
        if missing_mask.any():
            missing_smiles = aligned.index[missing_mask].tolist()
            preview = missing_smiles[:5]
            raise KeyError(
                f"Feature source {self.name!r} could not find {len(missing_smiles)} SMILES in {self.csv_path}. "
                f"Examples: {preview}"
            )

        return aligned.reset_index(drop=True)

    def describe(self) -> dict[str, object]:
        description = super().describe()
        description.update(
            {
                "csv_path": str(self.csv_path),
                "smiles_column": self.smiles_column,
                "feature_columns": None if self.feature_columns is None else list(self.feature_columns),
                "drop_columns": sorted(self.drop_columns),
                "prefix": self.prefix,
            }
        )
        return description


def build_csv_smiles_lookup_source(spec: dict[str, object]) -> CSVSmilesLookupSource:
    spec = dict(spec)
    spec.pop("source_type", None)
    spec.setdefault("name", "csv_smiles_lookup")
    return CSVSmilesLookupSource(**spec)


register_feature_source("csv_smiles_lookup", build_csv_smiles_lookup_source)

