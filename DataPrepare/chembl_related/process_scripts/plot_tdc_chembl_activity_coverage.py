#!/usr/bin/env python3
import csv
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
CHEMBL_DIR = ROOT / "DataPrepare/chembl_related"
DB = CHEMBL_DIR / "chembl_36/chembl_36_sqlite/chembl_36.db"
MAPPING_TSV = CHEMBL_DIR / "processed_data/chembl_tdc_overlap/tdc_unique_molecule_chembl_matches.tsv"
OUT_DIR = CHEMBL_DIR / "processed_data/chembl_tdc_activity_coverage"
MATCH_METHOD_PRIORITY = ("chembl_parent", "chembl_standardized", "rdkit")
ACTIVITY_THRESHOLD = 25


def truthy(value: str | None) -> bool:
    return str(value or "").lower() == "true"


def strict_match(row: dict[str, str]) -> tuple[str, str] | None:
    for method in MATCH_METHOD_PRIORITY:
        if not truthy(row.get(f"{method}_full_match")):
            continue
        chembl_ids = [item for item in row.get(f"{method}_chembl_ids", "").split(";") if item]
        if chembl_ids:
            return method, chembl_ids[0]
    return None


def activity_counts_by_chembl_id(conn: sqlite3.Connection, chembl_ids: list[str]) -> dict[str, int]:
    conn.execute("CREATE TEMP TABLE wanted_chembl_ids (chembl_id TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT OR IGNORE INTO wanted_chembl_ids (chembl_id) VALUES (?)",
        [(chembl_id,) for chembl_id in chembl_ids],
    )
    counts = {
        chembl_id: int(count)
        for chembl_id, count in conn.execute(
            """
            SELECT
              w.chembl_id,
              (
                SELECT COUNT(*)
                FROM activities act
                WHERE act.molregno = md.molregno
              ) AS n_activities
            FROM wanted_chembl_ids w
            JOIN molecule_dictionary md ON md.chembl_id = w.chembl_id
            """
        )
    }
    conn.execute("DROP TABLE wanted_chembl_ids")
    return counts


def bucket_activity_count(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 9:
        return "5-9"
    if value <= 24:
        return "10-24"
    if value <= 49:
        return "25-49"
    if value <= 99:
        return "50-99"
    if value <= 249:
        return "100-249"
    if value <= 499:
        return "250-499"
    if value <= 999:
        return "500-999"
    return "1000+"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    strict_rows = []
    with MAPPING_TSV.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(row)
            match = strict_match(row)
            if match is not None:
                method, chembl_id = match
                strict_rows.append({"smiles": row["smiles"], "match_method": method, "chembl_id": chembl_id})

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    counts = activity_counts_by_chembl_id(conn, sorted({row["chembl_id"] for row in strict_rows}))
    conn.close()

    output_rows = []
    for row in strict_rows:
        n_activities = counts.get(row["chembl_id"], 0)
        output_rows.append(
            {
                **row,
                "n_activities": n_activities,
                "include_activities_in_tool": n_activities < ACTIVITY_THRESHOLD,
            }
        )

    with (OUT_DIR / "tdc_strict_matched_activity_counts.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["smiles", "match_method", "chembl_id", "n_activities", "include_activities_in_tool"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    bucket_counts = Counter(bucket_activity_count(int(row["n_activities"])) for row in output_rows)
    bucket_order = ["0", "1", "2-4", "5-9", "10-24", "25-49", "50-99", "100-249", "250-499", "500-999", "1000+"]
    method_counts = Counter(row["match_method"] for row in output_rows)
    eligible = sum(1 for row in output_rows if row["include_activities_in_tool"])

    summary = {
        "mapping_tsv": str(MAPPING_TSV),
        "chembl_db": str(DB),
        "activity_threshold_rule": f"tool includes activities only when n_activities < {ACTIVITY_THRESHOLD}",
        "unique_tdc_smiles": len(rows),
        "strict_matched_unique_smiles": len(output_rows),
        "eligible_for_activity_details_unique_smiles": eligible,
        "eligible_pct_of_strict_matched": 100.0 * eligible / len(output_rows) if output_rows else 0.0,
        "strict_match_method_counts": dict(method_counts),
        "activity_count_buckets": {bucket: bucket_counts.get(bucket, 0) for bucket in bucket_order},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].bar(bucket_order, [bucket_counts.get(bucket, 0) for bucket in bucket_order], color="#4c78a8")
    axes[0].axvline(4.5, color="#c44e52", linestyle="--", linewidth=1.5)
    axes[0].set_title("Activity rows per strict matched TDC molecule")
    axes[0].set_xlabel("ChEMBL activity row count bucket")
    axes[0].set_ylabel("Unique TDC molecules")
    axes[0].tick_params(axis="x", rotation=35)

    labels = ["include activities", "properties only"]
    values = [eligible, len(output_rows) - eligible]
    axes[1].bar(labels, values, color=["#55a868", "#c44e52"])
    axes[1].set_title("chembl_info output policy")
    axes[1].set_ylabel("Unique TDC molecules")
    for idx, value in enumerate(values):
        axes[1].text(idx, value, f"{value:,}", ha="center", va="bottom")

    fig.savefig(OUT_DIR / "tdc_strict_matched_activity_coverage.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
