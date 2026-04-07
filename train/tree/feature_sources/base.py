from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class FeatureSource(ABC):
    def __init__(self, *, name: str):
        self.name = name

    @abstractmethod
    def load(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_type": self.__class__.__name__,
        }


class CompositeFeatureSource(FeatureSource):
    def __init__(self, sources: Sequence[FeatureSource]):
        source_names = [source.name for source in sources]
        super().__init__(name="+".join(source_names))
        self.sources = list(sources)

    def load(self, smiles_list: Sequence[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        used_columns: set[str] = set()
        for source in self.sources:
            frame = source.load(smiles_list).reset_index(drop=True)
            duplicated_columns = sorted(set(frame.columns) & used_columns)
            if duplicated_columns:
                raise ValueError(
                    f"Feature source {source.name!r} produced duplicated columns: {duplicated_columns[:10]}"
                )
            used_columns.update(frame.columns)
            frames.append(frame)
        if not frames:
            raise ValueError("CompositeFeatureSource received no feature sources")
        return pd.concat(frames, axis=1)

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_type": "CompositeFeatureSource",
            "sources": [source.describe() for source in self.sources],
        }

