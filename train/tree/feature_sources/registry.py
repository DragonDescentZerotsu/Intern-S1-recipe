from __future__ import annotations

import json
from pathlib import Path

from .base import CompositeFeatureSource, FeatureSource


FeatureSourceFactory = callable

_FEATURE_SOURCE_FACTORIES: dict[str, object] = {}


def register_feature_source(source_type: str, factory) -> None:
    if source_type in _FEATURE_SOURCE_FACTORIES:
        raise ValueError(f"Feature source {source_type!r} is already registered")
    _FEATURE_SOURCE_FACTORIES[source_type] = factory


def get_registered_feature_source_types() -> list[str]:
    return sorted(_FEATURE_SOURCE_FACTORIES)


def build_feature_source(spec: dict[str, object]) -> FeatureSource:
    source_type = spec.get("source_type")
    if not source_type:
        raise ValueError(f"Feature source spec is missing 'source_type': {spec}")
    try:
        factory = _FEATURE_SOURCE_FACTORIES[str(source_type)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature source {source_type!r}. "
            f"Known types: {get_registered_feature_source_types()}"
        ) from exc
    return factory(dict(spec))


def _normalize_feature_config(config_payload: object) -> tuple[list[dict[str, object]], str | None]:
    if isinstance(config_payload, dict):
        if "sources" in config_payload:
            sources = config_payload["sources"]
            if not isinstance(sources, list) or not sources:
                raise ValueError("Feature config 'sources' must be a non-empty list")
            return [dict(spec) for spec in sources], config_payload.get("feature_set_name")
        return [dict(config_payload)], config_payload.get("feature_set_name")
    if isinstance(config_payload, list) and config_payload:
        return [dict(spec) for spec in config_payload], None
    raise ValueError("Feature config must be a dict or non-empty list")


def load_feature_specs_from_paths(config_paths: list[str | Path]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    specs: list[dict[str, object]] = []
    loaded_configs: list[dict[str, object]] = []
    for path_like in config_paths:
        path = Path(path_like)
        with path.open("r", encoding="utf-8") as handle:
            config_payload = json.load(handle)
        source_specs, feature_set_name = _normalize_feature_config(config_payload)
        specs.extend(source_specs)
        loaded_configs.append(
            {
                "config_path": str(path.resolve()),
                "feature_set_name": feature_set_name,
                "sources": source_specs,
            }
        )
    if not specs:
        raise ValueError("No feature source specs were loaded")
    return specs, loaded_configs


def build_composite_feature_source(specs: list[dict[str, object]]) -> CompositeFeatureSource:
    return CompositeFeatureSource([build_feature_source(spec) for spec in specs])

