#!/usr/bin/env python3
import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "DataPrepare/chembl_related/chembl_36/chembl_36_sqlite/chembl_36.db"
OUT_DIR = ROOT / "outputs/chembl_activity_distribution"


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="nearest"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    total_molecules = conn.execute("SELECT COUNT(*) FROM molecule_dictionary").fetchone()[0]
    total_activities = conn.execute("SELECT COUNT(*) FROM activities").fetchone()[0]

    rows = conn.execute(
        """
        SELECT molregno, COUNT(*) AS n_activities
        FROM activities
        GROUP BY molregno
        ORDER BY molregno
        """
    ).fetchall()
    molregnos = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    counts = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))

    active_molecules = int(counts.size)
    zero_molecules = int(total_molecules - active_molecules)

    chembl25 = conn.execute(
        """
        SELECT md.chembl_id, md.pref_name, md.molregno,
               COUNT(a.activity_id) AS n_activities,
               COUNT(DISTINCT a.assay_id) AS n_assays
        FROM molecule_dictionary md
        LEFT JOIN activities a ON a.molregno = md.molregno
        WHERE md.chembl_id = 'CHEMBL25'
        GROUP BY md.chembl_id, md.pref_name, md.molregno
        """
    ).fetchone()

    top_rows = conn.execute(
        """
        SELECT md.chembl_id, COALESCE(md.pref_name, '') AS pref_name, x.molregno, x.n_activities
        FROM (
            SELECT molregno, COUNT(*) AS n_activities
            FROM activities
            GROUP BY molregno
            ORDER BY n_activities DESC
            LIMIT 20
        ) x
        JOIN molecule_dictionary md ON md.molregno = x.molregno
        ORDER BY x.n_activities DESC
        """
    ).fetchall()
    conn.close()

    bins = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            11,
            21,
            51,
            101,
            201,
            501,
            1001,
            2001,
            5001,
            10001,
            max(int(counts.max()) + 1, 10002),
        ],
        dtype=np.int64,
    )
    hist, edges = np.histogram(counts, bins=bins)
    bucket_labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == left + 1:
            bucket_labels.append(str(left))
        else:
            bucket_labels.append(f"{left}-{right - 1}")

    order = np.argsort(counts)
    sorted_counts = counts[order]
    ccdf_x = np.unique(sorted_counts)
    ccdf_y = active_molecules - np.searchsorted(sorted_counts, ccdf_x, side="left")

    quantiles = {
        "p50": percentile(counts, 50),
        "p75": percentile(counts, 75),
        "p90": percentile(counts, 90),
        "p95": percentile(counts, 95),
        "p99": percentile(counts, 99),
        "p99.5": percentile(counts, 99.5),
        "p99.9": percentile(counts, 99.9),
        "max": int(counts.max()),
    }

    thresholds = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
    threshold_counts = {f">={t}": int((counts >= t).sum()) for t in thresholds}

    summary = {
        "database": str(DB),
        "counting_rule": "Number of rows in activities per molregno; molecules with no activities counted as zero in totals.",
        "total_molecules_in_molecule_dictionary": int(total_molecules),
        "total_activity_rows": int(total_activities),
        "molecules_with_at_least_one_activity": active_molecules,
        "molecules_with_zero_activity": zero_molecules,
        "mean_activities_per_active_molecule": float(counts.mean()),
        "mean_activities_per_all_molecules": float(total_activities / total_molecules),
        "quantiles_active_molecules": quantiles,
        "threshold_counts_active_molecules": threshold_counts,
        "chembl25": {
            "chembl_id": chembl25[0],
            "pref_name": chembl25[1],
            "molregno": chembl25[2],
            "n_activities": chembl25[3],
            "n_distinct_assays": chembl25[4],
        },
        "top_20_by_activity_count": [
            {
                "chembl_id": row[0],
                "pref_name": row[1],
                "molregno": row[2],
                "n_activities": row[3],
            }
            for row in top_rows
        ],
        "histogram_active_molecules": [
            {"bucket": label, "n_molecules": int(value)}
            for label, value in zip(bucket_labels, hist)
        ],
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / "activity_count_histogram.tsv").open("w", encoding="utf-8") as handle:
        handle.write("bucket\tn_molecules\n")
        for label, value in zip(bucket_labels, hist):
            handle.write(f"{label}\t{int(value)}\n")
    with (OUT_DIR / "top_20_activity_counts.tsv").open("w", encoding="utf-8") as handle:
        handle.write("chembl_id\tpref_name\tmolregno\tn_activities\n")
        for row in top_rows:
            handle.write(f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\n")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    lefts = edges[:-1]
    widths = edges[1:] - edges[:-1]
    axes[0].bar(lefts, hist, width=widths, align="edge", color="#3b7ea1", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Activity rows per molecule")
    axes[0].set_ylabel("Number of molecules")
    axes[0].set_title("ChEMBL 36 activity count distribution")
    axes[0].axvline(chembl25[3], color="#c23b22", linestyle="--", linewidth=1.8)
    axes[0].text(
        chembl25[3] * 1.08,
        max(hist) * 0.35,
        f"CHEMBL25\n{chembl25[3]} activities",
        color="#8b1e12",
        va="top",
    )

    axes[1].plot(ccdf_x, ccdf_y / active_molecules, color="#3f6b3f", linewidth=2)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Activity rows per molecule >= x")
    axes[1].set_ylabel("Fraction of active molecules")
    axes[1].set_title("Tail view (CCDF)")
    axes[1].axvline(chembl25[3], color="#c23b22", linestyle="--", linewidth=1.8)
    axes[1].grid(True, which="both", alpha=0.25)

    subtitle = (
        f"Active molecules: {active_molecules:,} / {total_molecules:,}; "
        f"activity rows: {total_activities:,}; "
        f"median={quantiles['p50']:.0f}, p95={quantiles['p95']:.0f}, "
        f"p99={quantiles['p99']:.0f}, p99.9={quantiles['p99.9']:.0f}"
    )
    fig.suptitle(subtitle, y=1.03, fontsize=12)
    fig.savefig(OUT_DIR / "chembl36_activity_count_distribution.png", dpi=180, bbox_inches="tight")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
