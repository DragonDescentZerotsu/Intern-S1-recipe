#!/usr/bin/env python3
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
TDC_DIR = ROOT / "DataPrepare/TDC_no_conflict_labels_salt_removed"
DB = ROOT / "DataPrepare/chembl_related/chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_DIR = ROOT / "outputs/chembl_tdc_overlap"


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def rdkit_standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "inchi_key": Chem.MolToInchiKey(mol),
    }


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


def load_chembl_connectivity_blocks(conn):
    found = set()
    for (key,) in conn.execute("SELECT standard_inchi_key FROM compound_structures"):
        if key:
            found.add(key[:14])
    return found


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    unique_smiles = {}
    for path in sorted(TDC_DIR.glob("*/*.jsonl")):
        split = path.parent.name
        dataset = path.stem
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
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
                    continue
                unique_smiles.setdefault(smiles, rdkit_standardize(smiles))
                records.append(
                    {
                        "split": split,
                        "dataset": dataset,
                        "line_no": line_no,
                        "smiles": smiles,
                    }
                )

    for rec in records:
        std = unique_smiles.get(rec["smiles"])
        if std is None:
            rec.update(
                {
                    "valid_rdkit": False,
                    "canonical_smiles": "",
                    "inchi_key": "",
                    "inchi_key_connectivity": "",
                }
            )
        else:
            rec.update(
                {
                    "valid_rdkit": True,
                    "canonical_smiles": std["canonical_smiles"],
                    "inchi_key": std["inchi_key"],
                    "inchi_key_connectivity": std["inchi_key"][:14],
                }
            )

    valid_keys = {std["inchi_key"] for std in unique_smiles.values() if std is not None}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    full_found, rows_by_key = query_chembl_full_keys(conn, valid_keys)
    chembl_blocks = load_chembl_connectivity_blocks(conn)
    conn.close()

    unique_summary = {}
    for smiles, std in unique_smiles.items():
        if std is None:
            unique_summary[smiles] = {
                "smiles": smiles,
                "valid_rdkit": False,
                "canonical_smiles": "",
                "inchi_key": "",
                "inchi_key_connectivity": "",
                "chembl_full_inchikey_match": False,
                "chembl_connectivity_match": False,
                "chembl_ids": "",
            }
            continue
        key = std["inchi_key"]
        chembl_ids = sorted({row[1] for row in rows_by_key.get(key, [])})
        unique_summary[smiles] = {
            "smiles": smiles,
            "valid_rdkit": True,
            "canonical_smiles": std["canonical_smiles"],
            "inchi_key": key,
            "inchi_key_connectivity": key[:14],
            "chembl_full_inchikey_match": key in full_found,
            "chembl_connectivity_match": key[:14] in chembl_blocks,
            "chembl_ids": ";".join(chembl_ids),
        }

    per_dataset = defaultdict(Counter)
    overall = Counter()
    for rec in records:
        row = unique_summary.get(rec["smiles"])
        valid = bool(row and row["valid_rdkit"])
        full = bool(row and row["chembl_full_inchikey_match"])
        block = bool(row and row["chembl_connectivity_match"])
        for counter in (overall, per_dataset[(rec["split"], rec["dataset"])]):
            counter["rows"] += 1
            counter["valid_rdkit_rows"] += int(valid)
            counter["full_match_rows"] += int(full)
            counter["connectivity_match_rows"] += int(block)

    unique_counter = Counter()
    for row in unique_summary.values():
        unique_counter["unique_smiles"] += 1
        unique_counter["valid_rdkit_unique"] += int(row["valid_rdkit"])
        unique_counter["full_match_unique"] += int(row["chembl_full_inchikey_match"])
        unique_counter["connectivity_match_unique"] += int(row["chembl_connectivity_match"])

    unmatched_unique = [
        row
        for row in unique_summary.values()
        if row["valid_rdkit"] and not row["chembl_full_inchikey_match"]
    ]
    unmatched_unique.sort(key=lambda r: (not r["chembl_connectivity_match"], r["smiles"]))

    with (OUT_DIR / "tdc_unique_molecule_chembl_matches.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "smiles",
                "valid_rdkit",
                "canonical_smiles",
                "inchi_key",
                "inchi_key_connectivity",
                "chembl_full_inchikey_match",
                "chembl_connectivity_match",
                "chembl_ids",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(sorted(unique_summary.values(), key=lambda r: r["smiles"]))

    with (OUT_DIR / "tdc_unmatched_unique_molecules.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "smiles",
                "canonical_smiles",
                "inchi_key",
                "inchi_key_connectivity",
                "chembl_connectivity_match",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in unmatched_unique:
            writer.writerow(
                {
                    "smiles": row["smiles"],
                    "canonical_smiles": row["canonical_smiles"],
                    "inchi_key": row["inchi_key"],
                    "inchi_key_connectivity": row["inchi_key_connectivity"],
                    "chembl_connectivity_match": row["chembl_connectivity_match"],
                }
            )

    per_dataset_rows = []
    for (split, dataset), counter in sorted(per_dataset.items()):
        rows = counter["rows"]
        per_dataset_rows.append(
            {
                "split": split,
                "dataset": dataset,
                "rows": rows,
                "valid_rdkit_rows": counter["valid_rdkit_rows"],
                "full_match_rows": counter["full_match_rows"],
                "full_match_pct": 100.0 * counter["full_match_rows"] / rows,
                "connectivity_match_rows": counter["connectivity_match_rows"],
                "connectivity_match_pct": 100.0 * counter["connectivity_match_rows"] / rows,
            }
        )

    with (OUT_DIR / "tdc_chembl_match_by_dataset_split.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "dataset",
                "rows",
                "valid_rdkit_rows",
                "full_match_rows",
                "full_match_pct",
                "connectivity_match_rows",
                "connectivity_match_pct",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(per_dataset_rows)

    summary = {
        "tdc_dir": str(TDC_DIR),
        "chembl_db": str(DB),
        "match_rule": "RDKit MolToInchiKey(SMILES) matched against ChEMBL36 compound_structures.standard_inchi_key.",
        "connectivity_rule": "First 14 characters of InChIKey matched as a lenient connectivity-only check.",
        "rows": dict(overall),
        "unique": dict(unique_counter),
        "full_match_unique_pct": 100.0 * unique_counter["full_match_unique"] / unique_counter["unique_smiles"],
        "connectivity_match_unique_pct": 100.0
        * unique_counter["connectivity_match_unique"]
        / unique_counter["unique_smiles"],
        "full_unmatched_unique_valid": len(unmatched_unique),
        "full_unmatched_but_connectivity_matched_unique": sum(
            int(row["chembl_connectivity_match"]) for row in unmatched_unique
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
