from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOOLS_DIR = PROJECT_ROOT / "tools"
CHEMBL_DIR = PROJECT_ROOT / "DataPrepare/chembl_related"
DEFAULT_MAPPING_TSV = CHEMBL_DIR / "processed_data/chembl_tdc_overlap/tdc_unique_molecule_chembl_matches.tsv"
DEFAULT_OUTPUT_DIR = CHEMBL_DIR / "processed_data/chembl_tool_cache"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


_MODE = "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute chembl_info tool outputs for TDC molecules.")
    parser.add_argument("--mapping-tsv", type=Path, default=DEFAULT_MAPPING_TSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["both", "with_activities", "properties_only"],
        default="both",
        help="Which chembl_info cache file(s) to build.",
    )
    parser.add_argument("--num-workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def cache_path(output_dir: Path, include_activities: bool) -> Path:
    filename = "chembl_info_with_activities.jsonl" if include_activities else "chembl_info_properties_only.jsonl"
    return output_dir / filename


def load_smiles(mapping_tsv: Path, limit: int | None = None) -> list[str]:
    with mapping_tsv.open(encoding="utf-8", newline="") as handle:
        smiles = [row["smiles"] for row in csv.DictReader(handle, delimiter="\t")]
    if limit is not None:
        return smiles[:limit]
    return smiles


def init_worker(mode: str) -> None:
    global _MODE
    _MODE = mode


def compute_one(smiles: str) -> tuple[str, str | None, str | None]:
    from chembl_info import chembl_info

    with_activities = None
    properties_only = None
    if _MODE in {"both", "with_activities"}:
        with_activities = chembl_info(smiles, include_activities=True, use_cache=False)
    if _MODE in {"both", "properties_only"}:
        properties_only = chembl_info(smiles, include_activities=False, use_cache=False)
    return smiles, with_activities, properties_only


def open_outputs(output_dir: Path, mode: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {}
    if mode in {"both", "with_activities"}:
        path = cache_path(output_dir, True)
        handles[True] = (path, path.with_suffix(path.suffix + ".tmp").open("w", encoding="utf-8"))
    if mode in {"both", "properties_only"}:
        path = cache_path(output_dir, False)
        handles[False] = (path, path.with_suffix(path.suffix + ".tmp").open("w", encoding="utf-8"))
    return handles


def write_row(handle, smiles: str, result: str) -> None:
    handle.write(json.dumps({"smiles": smiles, "result": result}, ensure_ascii=False) + "\n")


def finalize_outputs(handles) -> None:
    for path, handle in handles.values():
        handle.close()
        path.with_suffix(path.suffix + ".tmp").replace(path)


def build_cache(smiles: Iterable[str], args: argparse.Namespace) -> None:
    smiles_list = list(smiles)
    handles = open_outputs(args.output_dir, args.mode)
    start_method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(start_method)

    try:
        with context.Pool(
            processes=min(args.num_workers, len(smiles_list) or 1),
            initializer=init_worker,
            initargs=(args.mode,),
        ) as pool:
            iterator = pool.imap_unordered(compute_one, smiles_list, chunksize=args.chunksize)
            for done, (smiles, with_activities, properties_only) in enumerate(iterator, start=1):
                if with_activities is not None:
                    write_row(handles[True][1], smiles, with_activities)
                if properties_only is not None:
                    write_row(handles[False][1], smiles, properties_only)
                if args.progress_every and done % args.progress_every == 0:
                    print(f"processed {done}/{len(smiles_list)}", flush=True)
    except Exception:
        for _, handle in handles.values():
            handle.close()
        raise

    finalize_outputs(handles)
    for include_activities, (path, _) in handles.items():
        mode_name = "with activities" if include_activities else "properties only"
        print(f"wrote {mode_name} cache: {path}", flush=True)


def main() -> None:
    args = parse_args()
    smiles = load_smiles(args.mapping_tsv, limit=args.limit)
    print(f"building chembl_info cache for {len(smiles)} SMILES with mode={args.mode}", flush=True)
    print(f"workers={args.num_workers} chunksize={args.chunksize}", flush=True)
    build_cache(smiles, args)


if __name__ == "__main__":
    main()
