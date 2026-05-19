#!/usr/bin/env python3
import argparse
import csv
import importlib.metadata
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem

try:
    from chembl_structure_pipeline import standardizer as chembl_standardizer
except ImportError:
    chembl_standardizer = None


ROOT = Path(__file__).resolve().parents[3]
CHEMBL_RELATED_DIR = ROOT / "DataPrepare/chembl_related"
TDC_DIR = ROOT / "DataPrepare/TDC_no_conflict_labels_salt_removed"
DB = CHEMBL_RELATED_DIR / "chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_DIR = CHEMBL_RELATED_DIR / "processed_data/chembl_tdc_overlap"
METHODS = ("rdkit", "chembl_standardized", "chembl_parent")
CONNECTIVITY_FULL_SCAN_PREFIX_THRESHOLD = 10000


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether molecules in the salt-removed TDC data can be found in "
            "ChEMBL compound_structures by InChIKey. The RDKit-only method is kept "
            "as a baseline; ChEMBL methods use chembl_structure_pipeline when installed."
        )
    )
    parser.add_argument("--tdc-dir", type=Path, default=TDC_DIR)
    parser.add_argument("--chembl-db", type=Path, default=DB)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Standardization methods to run.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test limit on loaded TDC rows. Use a separate --out-dir when set.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N newly standardized unique SMILES. Use 0 to disable.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(32, os.cpu_count() or 1),
        help="Number of worker processes for molecule standardization. Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=32,
        help="Multiprocessing chunksize for unique SMILES standardization.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be >= 1")
    needs_chembl = any(method.startswith("chembl_") for method in args.methods)
    if needs_chembl and chembl_standardizer is None:
        parser.error(
            "chembl_structure_pipeline is required for chembl_* methods. "
            "Install the ChEMBL 36 pipeline version with: "
            "python -m pip install chembl_structure_pipeline==1.2.0"
        )
    return args


def mol_summary(mol):
    key = Chem.MolToInchiKey(mol)
    return {
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "inchi_key": key,
        "inchi_key_connectivity": key[:14] if key else "",
        "error": "",
    }


def invalid_summary(error):
    return {
        "valid": False,
        "canonical_smiles": "",
        "inchi_key": "",
        "inchi_key_connectivity": "",
        "error": error,
    }


def mol_from_molblock(molblock):
    mol = Chem.MolFromMolBlock(molblock, sanitize=True, removeHs=False)
    if mol is not None:
        return mol
    mol = Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return mol


def standardize_smiles(smiles, methods):
    summaries = {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid = invalid_summary("rdkit_parse_failed")
        return {method: dict(invalid) for method in methods}

    for method in methods:
        if method == "rdkit":
            summaries[method] = mol_summary(mol)
        elif method not in ("chembl_standardized", "chembl_parent"):
            raise ValueError(f"Unknown method: {method}")

    if any(method.startswith("chembl_") for method in methods):
        try:
            molblock = Chem.MolToMolBlock(mol)
            std_molblock = chembl_standardizer.standardize_molblock(molblock)
            if "chembl_standardized" in methods:
                std_mol = mol_from_molblock(std_molblock)
                summaries["chembl_standardized"] = (
                    mol_summary(std_mol)
                    if std_mol is not None
                    else invalid_summary("chembl_pipeline_parse_failed")
                )
            if "chembl_parent" in methods:
                parent_result = chembl_standardizer.get_parent_molblock(std_molblock)
                parent_molblock = parent_result[0] if isinstance(parent_result, tuple) else parent_result
                parent_mol = mol_from_molblock(parent_molblock)
                summaries["chembl_parent"] = (
                    mol_summary(parent_mol)
                    if parent_mol is not None
                    else invalid_summary("chembl_parent_parse_failed")
                )
        except Exception as exc:
            invalid = invalid_summary(f"{type(exc).__name__}: {exc}")
            for method in methods:
                if method.startswith("chembl_"):
                    summaries[method] = dict(invalid)
    return summaries


def standardize_one_smiles(task):
    smiles, methods = task
    return smiles, standardize_smiles(smiles, methods)


def standardize_unique_smiles(smiles_list, methods, num_workers, chunksize, progress_every):
    total = len(smiles_list)
    unique_smiles = {}
    started = time.monotonic()

    def maybe_print_progress(done):
        if not progress_every:
            return
        if done % progress_every == 0 or done == total:
            elapsed = time.monotonic() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            print(
                f"standardized_unique_smiles={done}/{total} "
                f"elapsed_sec={elapsed:.1f} rate_per_sec={rate:.2f}",
                file=sys.stderr,
                flush=True,
            )

    if num_workers == 1 or total <= 1:
        for done, smiles in enumerate(smiles_list, start=1):
            unique_smiles[smiles] = standardize_smiles(smiles, methods)
            maybe_print_progress(done)
        return unique_smiles

    ctx = mp.get_context("fork")
    tasks = ((smiles, methods) for smiles in smiles_list)
    with ctx.Pool(processes=num_workers) as pool:
        for done, (smiles, summaries) in enumerate(
            pool.imap_unordered(standardize_one_smiles, tasks, chunksize=chunksize),
            start=1,
        ):
            unique_smiles[smiles] = summaries
            maybe_print_progress(done)
    return unique_smiles


def query_chembl_full_keys(conn, keys):
    found = set()
    rows_by_key = defaultdict(list)
    keys = sorted(k for k in keys if k)
    for batch in chunks(keys, 800):
        placeholders = ",".join("?" for _ in batch)
        query = f"""
            SELECT cs.standard_inchi_key, cs.molregno, md.chembl_id, md.pref_name
            FROM compound_structures cs
            JOIN molecule_dictionary md ON md.molregno = cs.molregno
            WHERE cs.standard_inchi_key IN ({placeholders})
        """
        for key, molregno, chembl_id, pref_name in conn.execute(query, batch):
            found.add(key)
            rows_by_key[key].append((molregno, chembl_id, pref_name or ""))
    return found, rows_by_key


def query_chembl_connectivity_blocks(conn, prefixes):
    found = set()
    prefixes = sorted(prefix for prefix in prefixes if prefix)
    if len(prefixes) > CONNECTIVITY_FULL_SCAN_PREFIX_THRESHOLD:
        wanted = set(prefixes)
        for (key,) in conn.execute("SELECT standard_inchi_key FROM compound_structures"):
            if key and key[:14] in wanted:
                found.add(key[:14])
        return found

    for batch in chunks(prefixes, 100):
        clauses = []
        params = []
        for prefix in batch:
            clauses.append("(standard_inchi_key >= ? AND standard_inchi_key < ?)")
            params.extend([prefix, prefix + "{"])
        query = f"""
            SELECT DISTINCT substr(standard_inchi_key, 1, 14)
            FROM compound_structures
            WHERE {" OR ".join(clauses)}
        """
        for (prefix,) in conn.execute(query, params):
            if prefix:
                found.add(prefix)
    return found


def get_chembl_pipeline_version():
    if chembl_standardizer is None:
        return None
    try:
        return importlib.metadata.version("chembl_structure_pipeline")
    except importlib.metadata.PackageNotFoundError:
        return "installed_unknown_version"


def get_chembl_db_versions(conn):
    rows = []
    for row in conn.execute("SELECT * FROM version"):
        cells = ["" if cell is None else str(cell) for cell in row]
        if cells:
            rows.append(cells)
    return rows


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def write_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    unique_smiles_seen = set()
    unique_smiles_list = []
    loaded_records = 0
    for path in sorted(args.tdc_dir.glob("*/*.jsonl")):
        split = path.parent.name
        dataset = path.stem
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if args.max_records is not None and loaded_records >= args.max_records:
                    break
                item = json.loads(line)
                smiles = item.get("drug")
                if not smiles:
                    records.append(
                        {
                            "split": split,
                            "dataset": dataset,
                            "line_no": line_no,
                            "smiles": "",
                            "valid_rdkit": False,
                        }
                    )
                    loaded_records += 1
                    continue
                if smiles not in unique_smiles_seen:
                    unique_smiles_seen.add(smiles)
                    unique_smiles_list.append(smiles)
                records.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "line_no": line_no,
                        "smiles": smiles,
                    }
                )
                loaded_records += 1
            if args.max_records is not None and loaded_records >= args.max_records:
                break

    print(
        f"loaded_records={len(records)} unique_smiles={len(unique_smiles_list)} "
        f"num_workers={args.num_workers}",
        file=sys.stderr,
        flush=True,
    )
    unique_smiles = standardize_unique_smiles(
        unique_smiles_list,
        tuple(args.methods),
        args.num_workers,
        args.chunksize,
        args.progress_every,
    )

    valid_keys = {
        summary["inchi_key"]
        for summaries in unique_smiles.values()
        for summary in summaries.values()
        if summary["valid"] and summary["inchi_key"]
    }
    valid_prefixes = {key[:14] for key in valid_keys if key}
    conn = sqlite3.connect(f"file:{args.chembl_db}?mode=ro", uri=True)
    full_found, rows_by_key = query_chembl_full_keys(conn, valid_keys)
    chembl_blocks = query_chembl_connectivity_blocks(conn, valid_prefixes)
    db_versions = get_chembl_db_versions(conn)
    conn.close()

    unique_summary = {}
    for smiles, summaries in unique_smiles.items():
        row = {"smiles": smiles}
        for method in args.methods:
            summary = summaries[method]
            key = summary["inchi_key"]
            chembl_ids = sorted({chembl_row[1] for chembl_row in rows_by_key.get(key, [])})
            row.update(
                {
                    f"{method}_valid": summary["valid"],
                    f"{method}_canonical_smiles": summary["canonical_smiles"],
                    f"{method}_inchi_key": key,
                    f"{method}_inchi_key_connectivity": summary["inchi_key_connectivity"],
                    f"{method}_full_match": key in full_found if key else False,
                    f"{method}_connectivity_match": summary["inchi_key_connectivity"] in chembl_blocks
                    if summary["inchi_key_connectivity"]
                    else False,
                    f"{method}_chembl_ids": ";".join(chembl_ids),
                    f"{method}_error": summary["error"],
                }
            )

        if "rdkit" in args.methods:
            row.update(
                {
                    "valid_rdkit": row["rdkit_valid"],
                    "canonical_smiles": row["rdkit_canonical_smiles"],
                    "inchi_key": row["rdkit_inchi_key"],
                    "inchi_key_connectivity": row["rdkit_inchi_key_connectivity"],
                    "chembl_full_inchikey_match": row["rdkit_full_match"],
                    "chembl_connectivity_match": row["rdkit_connectivity_match"],
                    "chembl_ids": row["rdkit_chembl_ids"],
                }
            )
        unique_summary[smiles] = row

    per_dataset = defaultdict(Counter)
    overall = Counter()
    for rec in records:
        row = unique_summary.get(rec["smiles"])
        for counter in (overall, per_dataset[(rec["split"], rec["dataset"])]):
            counter["rows"] += 1
            for method in args.methods:
                valid = bool(row and row[f"{method}_valid"])
                full = bool(row and row[f"{method}_full_match"])
                block = bool(row and row[f"{method}_connectivity_match"])
                counter[f"{method}_valid_rows"] += int(valid)
                counter[f"{method}_full_match_rows"] += int(full)
                counter[f"{method}_connectivity_match_rows"] += int(block)

    unique_counters = {}
    unmatched_by_method = {}
    for method in args.methods:
        counter = Counter()
        unmatched = []
        for row in unique_summary.values():
            counter["unique_smiles"] += 1
            counter["valid_unique"] += int(row[f"{method}_valid"])
            counter["full_match_unique"] += int(row[f"{method}_full_match"])
            counter["connectivity_match_unique"] += int(row[f"{method}_connectivity_match"])
            if row[f"{method}_valid"] and not row[f"{method}_full_match"]:
                unmatched.append(row)
        unmatched.sort(key=lambda r: (not r[f"{method}_connectivity_match"], r["smiles"]))
        unique_counters[method] = counter
        unmatched_by_method[method] = unmatched

    method_fields = []
    for method in args.methods:
        method_fields.extend(
            [
                f"{method}_valid",
                f"{method}_canonical_smiles",
                f"{method}_inchi_key",
                f"{method}_inchi_key_connectivity",
                f"{method}_full_match",
                f"{method}_connectivity_match",
                f"{method}_chembl_ids",
                f"{method}_error",
            ]
        )
    legacy_fields = [
        "valid_rdkit",
        "canonical_smiles",
        "inchi_key",
        "inchi_key_connectivity",
        "chembl_full_inchikey_match",
        "chembl_connectivity_match",
        "chembl_ids",
    ]
    match_fields = ["smiles"]
    if "rdkit" in args.methods:
        match_fields.extend(legacy_fields)
    match_fields.extend(method_fields)
    write_tsv(
        out_dir / "tdc_unique_molecule_chembl_matches.tsv",
        sorted(unique_summary.values(), key=lambda r: r["smiles"]),
        match_fields,
    )

    for method, unmatched in unmatched_by_method.items():
        unmatched_rows = [
            {
                "smiles": row["smiles"],
                "canonical_smiles": row[f"{method}_canonical_smiles"],
                "inchi_key": row[f"{method}_inchi_key"],
                "inchi_key_connectivity": row[f"{method}_inchi_key_connectivity"],
                "chembl_connectivity_match": row[f"{method}_connectivity_match"],
                "error": row[f"{method}_error"],
            }
            for row in unmatched
        ]
        unmatched_path = out_dir / f"tdc_unmatched_unique_molecules_{method}.tsv"
        write_tsv(
            unmatched_path,
            unmatched_rows,
            [
                "smiles",
                "canonical_smiles",
                "inchi_key",
                "inchi_key_connectivity",
                "chembl_connectivity_match",
                "error",
            ],
        )
        if method == "rdkit":
            write_tsv(
                out_dir / "tdc_unmatched_unique_molecules.tsv",
                [
                    {key: row[key] for key in row if key != "error"}
                    for row in unmatched_rows
                ],
                [
                    "smiles",
                    "canonical_smiles",
                    "inchi_key",
                    "inchi_key_connectivity",
                    "chembl_connectivity_match",
                ],
            )

    per_dataset_rows = []
    for (split, dataset), counter in sorted(per_dataset.items()):
        rows = counter["rows"]
        dataset_row = {"split": split, "dataset": dataset, "rows": rows}
        for method in args.methods:
            dataset_row.update(
                {
                    f"{method}_valid_rows": counter[f"{method}_valid_rows"],
                    f"{method}_full_match_rows": counter[f"{method}_full_match_rows"],
                    f"{method}_full_match_pct": pct(counter[f"{method}_full_match_rows"], rows),
                    f"{method}_connectivity_match_rows": counter[f"{method}_connectivity_match_rows"],
                    f"{method}_connectivity_match_pct": pct(
                        counter[f"{method}_connectivity_match_rows"], rows
                    ),
                }
            )
        if "rdkit" in args.methods:
            dataset_row.update(
                {
                    "valid_rdkit_rows": counter["rdkit_valid_rows"],
                    "full_match_rows": counter["rdkit_full_match_rows"],
                    "full_match_pct": pct(counter["rdkit_full_match_rows"], rows),
                    "connectivity_match_rows": counter["rdkit_connectivity_match_rows"],
                    "connectivity_match_pct": pct(counter["rdkit_connectivity_match_rows"], rows),
                }
            )
        per_dataset_rows.append(dataset_row)

    per_dataset_fields = ["split", "dataset", "rows"]
    if "rdkit" in args.methods:
        per_dataset_fields.extend(
            [
                "valid_rdkit_rows",
                "full_match_rows",
                "full_match_pct",
                "connectivity_match_rows",
                "connectivity_match_pct",
            ]
        )
    for method in args.methods:
        per_dataset_fields.extend(
            [
                f"{method}_valid_rows",
                f"{method}_full_match_rows",
                f"{method}_full_match_pct",
                f"{method}_connectivity_match_rows",
                f"{method}_connectivity_match_pct",
            ]
        )
    write_tsv(out_dir / "tdc_chembl_match_by_dataset_split.tsv", per_dataset_rows, per_dataset_fields)

    method_summaries = {}
    for method in args.methods:
        unique_counter = unique_counters[method]
        method_summaries[method] = {
            "rows": {
                "rows": overall["rows"],
                "valid_rows": overall[f"{method}_valid_rows"],
                "full_match_rows": overall[f"{method}_full_match_rows"],
                "connectivity_match_rows": overall[f"{method}_connectivity_match_rows"],
            },
            "unique": dict(unique_counter),
            "full_match_unique_pct": pct(
                unique_counter["full_match_unique"], unique_counter["unique_smiles"]
            ),
            "connectivity_match_unique_pct": pct(
                unique_counter["connectivity_match_unique"], unique_counter["unique_smiles"]
            ),
            "full_unmatched_unique_valid": len(unmatched_by_method[method]),
            "full_unmatched_but_connectivity_matched_unique": sum(
                int(row[f"{method}_connectivity_match"]) for row in unmatched_by_method[method]
            ),
        }

    summary = {
        "tdc_dir": str(args.tdc_dir),
        "chembl_db": str(args.chembl_db),
        "methods": args.methods,
        "match_rule": "Method-specific InChIKeys matched against ChEMBL36 compound_structures.standard_inchi_key.",
        "connectivity_rule": "First 14 characters of InChIKey matched as a lenient connectivity-only check.",
        "chembl_structure_pipeline_version": get_chembl_pipeline_version(),
        "chembl_db_version_rows": db_versions,
        "method_summaries": method_summaries,
    }
    if "rdkit" in args.methods:
        rdkit_summary = method_summaries["rdkit"]
        summary.update(
            {
                "rows": {
                    "rows": rdkit_summary["rows"]["rows"],
                    "valid_rdkit_rows": rdkit_summary["rows"]["valid_rows"],
                    "full_match_rows": rdkit_summary["rows"]["full_match_rows"],
                    "connectivity_match_rows": rdkit_summary["rows"]["connectivity_match_rows"],
                },
                "unique": {
                    "unique_smiles": rdkit_summary["unique"]["unique_smiles"],
                    "valid_rdkit_unique": rdkit_summary["unique"]["valid_unique"],
                    "full_match_unique": rdkit_summary["unique"]["full_match_unique"],
                    "connectivity_match_unique": rdkit_summary["unique"][
                        "connectivity_match_unique"
                    ],
                },
                "full_match_unique_pct": rdkit_summary["full_match_unique_pct"],
                "connectivity_match_unique_pct": rdkit_summary["connectivity_match_unique_pct"],
                "full_unmatched_unique_valid": rdkit_summary["full_unmatched_unique_valid"],
                "full_unmatched_but_connectivity_matched_unique": rdkit_summary[
                    "full_unmatched_but_connectivity_matched_unique"
                ],
            }
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
