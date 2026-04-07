from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from itertools import repeat
import math
import os
from pathlib import Path
import sys

THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for env_name, env_value in THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(env_name, env_value)

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/local/tmp/matplotlib")

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from tools.pka_related_tools_for_feature_extract import estimate_logd_structured, predict_pka_structured

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


RDLogger.DisableLog("rdApp.*")

DEFAULT_INPUT_DIR = PROJECT_DIR / "DataPrepare" / "TDC_no_conflict_labels_salt_removed"
DEFAULT_OUTPUT_PATH = (
    PROJECT_DIR
    / "DataPrepare"
    / "mol_features_for_tree"
    / "rdkit_descriptors_and_pka"
    / "tdc_no_conflict_labels_salt_removed_unique_smiles_rdkit_descriptors_and_pka.csv"
)
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_PATH.with_suffix(".metadata.json")
DEFAULT_LOGD_PH = 7.4
DEFAULT_NUM_WORKERS = 1
DESCRIPTOR_FUNCS = list(Descriptors._descList)


def iter_unique_smiles_from_jsonl_dir(input_dir: Path) -> list[str]:
    unique_smiles: dict[str, None] = {}

    for jsonl_path in sorted(input_dir.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                smiles = str(record.get("drug", "")).strip()
                if smiles:
                    unique_smiles.setdefault(smiles, None)

    return list(unique_smiles)


def load_smiles_list(smiles_json_path: Path | None, input_dir: Path) -> list[str]:
    if smiles_json_path is None:
        return iter_unique_smiles_from_jsonl_dir(input_dir)

    with smiles_json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list of SMILES at {smiles_json_path}")

    smiles_list: list[str] = []
    for item in payload:
        smiles = str(item).strip()
        if smiles:
            smiles_list.append(smiles)
    if not smiles_list:
        raise ValueError(f"No SMILES were loaded from {smiles_json_path}")
    return smiles_list


def _safe_float(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isinf(numeric):
        return math.nan
    return numeric


def _series_stats(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_sum": 0.0,
            f"{prefix}_range": math.nan,
        }

    series = pd.Series(values, dtype="float64")
    return {
        f"{prefix}_min": _safe_float(series.min()),
        f"{prefix}_max": _safe_float(series.max()),
        f"{prefix}_mean": _safe_float(series.mean()),
        f"{prefix}_std": _safe_float(series.std(ddof=0)),
        f"{prefix}_sum": _safe_float(series.sum()),
        f"{prefix}_range": _safe_float(series.max() - series.min()),
    }


def _warning_flags(warnings: list[str]) -> dict[str, int]:
    return {
        "pka__warning_count": len(warnings),
        "pka__warning_multiple_basic_sites": int(
            any("Multiple basic sites" in warning for warning in warnings)
        ),
        "pka__warning_multiple_acidic_sites": int(
            any("Multiple acidic sites" in warning for warning in warnings)
        ),
        "pka__warning_amphoteric": int(
            any("Amphoteric molecule" in warning for warning in warnings)
        ),
        "pka__warning_fraction_neutral_clamped": int(
            any("non-positive" in warning for warning in warnings)
        ),
        "pka__warning_fraction_neutral_tiny": int(
            any("extremely small" in warning for warning in warnings)
        ),
    }


def _descriptor_features(mol: Chem.Mol) -> dict[str, float]:
    features: dict[str, float] = {}
    for descriptor_name, descriptor_fn in DESCRIPTOR_FUNCS:
        try:
            value = descriptor_fn(mol)
        except Exception:
            value = math.nan
        features[f"rdkit__{descriptor_name}"] = _safe_float(value)
    return features


def _pka_feature_record(smiles: str, *, logd_ph: float) -> tuple[dict[str, float | int | str | list[float]], int, int]:
    pka_result = predict_pka_structured(smiles)
    logd_result = estimate_logd_structured(smiles, ph=logd_ph, pka_result=pka_result)

    base_values = sorted((_safe_float(value) for value in pka_result["base_sites"].values()), reverse=True)
    acid_values = sorted(_safe_float(value) for value in pka_result["acid_sites"].values())

    features: dict[str, float | int | str | list[float]] = {
        "smiles": smiles,
        "pka__num_basic_sites": pka_result["num_basic_sites"],
        "pka__num_acidic_sites": pka_result["num_acidic_sites"],
        "pka__num_ionizable_sites": pka_result["num_basic_sites"] + pka_result["num_acidic_sites"],
        "pka__has_basic_site": int(pka_result["num_basic_sites"] > 0),
        "pka__has_acidic_site": int(pka_result["num_acidic_sites"] > 0),
        "pka__is_amphoteric": int(
            pka_result["num_basic_sites"] > 0 and pka_result["num_acidic_sites"] > 0
        ),
        "pka__most_basic_pka": _safe_float(pka_result["most_basic_pka"]),
        "pka__most_acidic_pka": _safe_float(pka_result["most_acidic_pka"]),
        "pka__logd_ph": _safe_float(logd_result["ph"]),
        "pka__logp_wildman_crippen": _safe_float(logd_result["logp"]),
        "pka__logd_estimate": _safe_float(logd_result["logd"]),
        "pka__fraction_neutral": _safe_float(logd_result["fraction_neutral"]),
        "_base_values": base_values,
        "_acid_values": acid_values,
    }
    features.update(_series_stats(base_values, "pka__base_site_pka"))
    features.update(_series_stats(acid_values, "pka__acid_site_pka"))
    features.update(_warning_flags(logd_result["warnings"]))
    return features, len(base_values), len(acid_values)


def _compute_smiles_feature_row(smiles: str, logd_ph: float) -> tuple[dict[str, object], int, int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES encountered in dataset: {smiles!r}")

    row: dict[str, object] = {"smiles": smiles}
    row.update(_descriptor_features(mol))

    pka_row, basic_count, acidic_count = _pka_feature_record(smiles, logd_ph=logd_ph)
    row.update(pka_row)
    return row, basic_count, acidic_count


def _iter_feature_rows(
    smiles_list: list[str],
    *,
    logd_ph: float,
    num_workers: int,
):
    if num_workers == 1:
        yield from (_compute_smiles_feature_row(smiles, logd_ph) for smiles in smiles_list)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        yield from executor.map(_compute_smiles_feature_row, smiles_list, repeat(logd_ph))


def build_feature_table(
    input_dir: Path,
    *,
    logd_ph: float = DEFAULT_LOGD_PH,
    num_workers: int = DEFAULT_NUM_WORKERS,
    smiles_limit: int | None = None,
    smiles_json_path: Path | None = None,
    progress_desc: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1 (got {num_workers})")

    unique_smiles = load_smiles_list(smiles_json_path, input_dir)
    if smiles_limit is not None:
        unique_smiles = unique_smiles[:smiles_limit]

    rows: list[dict[str, object]] = []
    max_basic_sites = 0
    max_acidic_sites = 0

    iterator = unique_smiles
    progress = None
    if tqdm is not None:
        progress = tqdm(
            unique_smiles,
            total=len(unique_smiles),
            desc=progress_desc or "feature extraction",
            dynamic_ncols=True,
            leave=True,
        )
        iterator = progress

    result_iterator = _iter_feature_rows(
        unique_smiles,
        logd_ph=logd_ph,
        num_workers=num_workers,
    )
    if progress is not None:
        iterator = zip(iterator, result_iterator)
    else:
        iterator = result_iterator

    for index, item in enumerate(iterator, start=1):
        if progress is not None:
            _, (row, basic_count, acidic_count) = item
        else:
            row, basic_count, acidic_count = item
        rows.append(row)
        max_basic_sites = max(max_basic_sites, basic_count)
        max_acidic_sites = max(max_acidic_sites, acidic_count)

        if progress is not None and index % 25 == 0:
            progress.set_postfix(
                max_basic=max_basic_sites,
                max_acidic=max_acidic_sites,
                refresh=False,
            )

    expanded_rows: list[dict[str, object]] = []
    for row in rows:
        row = dict(row)
        base_values = list(row.pop("_base_values"))
        acid_values = list(row.pop("_acid_values"))

        for rank in range(max_basic_sites):
            value = base_values[rank] if rank < len(base_values) else math.nan
            row[f"pka__base_site_pka_rank_{rank + 1}"] = value

        for rank in range(max_acidic_sites):
            value = acid_values[rank] if rank < len(acid_values) else math.nan
            row[f"pka__acid_site_pka_rank_{rank + 1}"] = value

        expanded_rows.append(row)

    frame = pd.DataFrame(expanded_rows)
    descriptor_names = [name for name, _ in DESCRIPTOR_FUNCS]
    metadata = {
        "input_dir": str(input_dir.resolve()),
        "smiles_json_path": None if smiles_json_path is None else str(smiles_json_path.resolve()),
        "unique_smiles": len(unique_smiles),
        "smiles_limit": smiles_limit,
        "logd_ph": logd_ph,
        "num_workers": num_workers,
        "rdkit_descriptor_count": len(descriptor_names),
        "rdkit_descriptor_names": descriptor_names,
        "max_basic_sites": max_basic_sites,
        "max_acidic_sites": max_acidic_sites,
        "feature_column_count": len(frame.columns) - 1,
    }
    return frame, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a unique-SMILES lookup table containing all RDKit descriptors plus "
            "numeric pKa/logD-derived features for train/tree."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--logd-ph",
        type=float,
        default=DEFAULT_LOGD_PH,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--smiles-limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--smiles-json-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--progress-desc",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, metadata = build_feature_table(
        args.input_dir,
        logd_ph=args.logd_ph,
        num_workers=args.num_workers,
        smiles_limit=args.smiles_limit,
        smiles_json_path=args.smiles_json_path,
        progress_desc=args.progress_desc,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, index=False)

    metadata = {
        **metadata,
        "output_path": str(args.output_path.resolve()),
        "metadata_path": str(args.metadata_path.resolve()),
    }
    with args.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"input_dir={args.input_dir}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"metadata_path={args.metadata_path}", flush=True)
    print(f"unique_smiles={metadata['unique_smiles']}", flush=True)
    print(f"feature_column_count={metadata['feature_column_count']}", flush=True)
    print(f"max_basic_sites={metadata['max_basic_sites']}", flush=True)
    print(f"max_acidic_sites={metadata['max_acidic_sites']}", flush=True)


if __name__ == "__main__":
    main()
