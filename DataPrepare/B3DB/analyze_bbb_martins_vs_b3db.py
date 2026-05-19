#!/usr/bin/env python3
"""Audit TDC BBB_Martins against B3DB and export filtered datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")

SPLITS = ("train", "valid", "test")
PROMPT_RE = re.compile(r"Drug SMILES:\s*(?:<SMILES>)?([^\n<]+)(?:</SMILES>)?")


@dataclass
class CanonicalResult:
    valid: bool
    exact: str = ""
    salt_removed: str = ""
    error: str = ""
    n_fragments: int = 0
    salt_removed_changed: bool = False


@dataclass
class Row:
    source: str
    split: str
    row_index: int
    smiles: str
    y: int
    raw: dict[str, Any]
    canon: CanonicalResult
    row_id: str = ""
    flags: list[str] = field(default_factory=list)
    b3db_exact_labels: str = ""
    b3db_salt_labels: str = ""
    b3db_exact_row_nos: str = ""
    b3db_exact_compound_names: str = ""
    b3db_exact_smiles: str = ""
    b3db_salt_row_nos: str = ""
    b3db_salt_compound_names: str = ""
    b3db_salt_smiles: str = ""
    tdc_exact_group_row_ids: str = ""
    tdc_salt_group_row_ids: str = ""
    b3db_match_type: str = "unmatched"


def canonicalize(smiles: str) -> CanonicalResult:
    if not isinstance(smiles, str) or not smiles.strip():
        return CanonicalResult(False, error="empty_smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return CanonicalResult(False, error="rdkit_parse_failed")

    try:
        exact = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        n_fragments = len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))
        chooser = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        cleaned = chooser.choose(mol)
        if cleaned is None:
            return CanonicalResult(False, exact=exact, error="largest_fragment_failed")
        salt_removed = Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)
        return CanonicalResult(
            True,
            exact=exact,
            salt_removed=salt_removed,
            n_fragments=n_fragments,
            salt_removed_changed=(exact != salt_removed),
        )
    except Exception as exc:  # noqa: BLE001 - keep row-level audit instead of aborting.
        return CanonicalResult(False, error=type(exc).__name__ + ": " + str(exc))


def extract_smiles_from_prompt(text: str) -> str:
    match = PROMPT_RE.search(text)
    if not match:
        raise ValueError("Drug SMILES field not found")
    return match.group(1).strip().strip("'\"")


def replace_prompt_smiles(text: str, new_smiles: str) -> str:
    match = PROMPT_RE.search(text)
    if not match:
        return text
    start, end = match.span(1)
    return text[:start] + new_smiles + text[end:]


def load_prompt_rows(base_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for split in SPLITS:
        path = base_dir / split / "BBB_Martins.jsonl"
        if not path.exists():
            path = base_dir.parent / f"TDC_{split}_prompts_label_scaffold" / "BBB_Martins.jsonl"
        with path.open() as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                smiles = extract_smiles_from_prompt(obj["text"])
                y = int(obj["Y"])
                canon = canonicalize(smiles)
                rows.append(
                    Row(
                        source="tdc_original_prompt",
                        split=split,
                        row_index=idx,
                        smiles=smiles,
                        y=y,
                        raw=obj,
                        canon=canon,
                        row_id=f"{split}:{idx}",
                    )
                )
    return rows


def load_salt_removed_rows(base_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for split in SPLITS:
        path = base_dir / split / "BBB_Martins.jsonl"
        with path.open() as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                smiles = obj["drug"]
                y = int(obj["Y"])
                canon = canonicalize(smiles)
                rows.append(
                    Row(
                        source="tdc_no_conflict_salt_removed",
                        split=split,
                        row_index=idx,
                        smiles=smiles,
                        y=y,
                        raw=obj,
                        canon=canon,
                        row_id=f"{split}:{idx}",
                    )
                )
    return rows


def label_to_int(label: Any) -> int:
    text = str(label).strip()
    if text == "BBB+":
        return 1
    if text == "BBB-":
        return 0
    raise ValueError(f"Unknown B3DB label: {label!r}")


def load_b3db(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        df.columns = [str(col).lstrip("\ufeff") for col in df.columns]
        source_name = path.stem
        for _, rec in df.iterrows():
            smiles = str(rec["SMILES"]).strip()
            canon = canonicalize(smiles)
            rows.append(
                {
                    "source_file": source_name,
                    "NO.": rec.get("NO."),
                    "compound_name": rec.get("compound_name"),
                    "SMILES": smiles,
                    "label": label_to_int(rec["BBB+/BBB-"]),
                    "label_text": str(rec["BBB+/BBB-"]),
                    "canon": canon,
                }
            )

    exact_map = build_label_map(rows, "exact")
    salt_map = build_label_map(rows, "salt_removed")
    meta = {
        "n_rows": len(rows),
        "n_valid_rdkit": sum(1 for r in rows if r["canon"].valid),
        "n_invalid_rdkit": sum(1 for r in rows if not r["canon"].valid),
        "source_files": [str(p) for p in paths if p.exists()],
        "exact_conflict_keys": count_conflict_keys(exact_map),
        "salt_removed_conflict_keys": count_conflict_keys(salt_map),
        "exact_map": exact_map,
        "salt_map": salt_map,
    }
    return rows, meta


def build_label_map(rows: list[dict[str, Any]], attr: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        canon: CanonicalResult = row["canon"]
        if not canon.valid:
            continue
        key = getattr(canon, attr)
        entry = out.setdefault(
            key,
            {"labels": set(), "row_nos": [], "smiles": [], "compound_names": []},
        )
        entry["labels"].add(int(row["label"]))
        entry["row_nos"].append(f"{row['source_file']}:{row['NO.']}")
        entry["smiles"].append(row["SMILES"])
        entry["compound_names"].append(str(row["compound_name"]))
    return out


def count_conflict_keys(label_map: dict[str, dict[str, Any]]) -> int:
    return sum(1 for entry in label_map.values() if len(entry["labels"]) > 1)


def labels_as_text(labels: set[int]) -> str:
    if not labels:
        return ""
    return "|".join(str(v) for v in sorted(labels))


def add_group_flags(rows: list[Row], key_attr: str, conflict_flag: str, duplicate_flag: str) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        if row.canon.valid:
            groups[getattr(row.canon, key_attr)].append(row)

    for group_rows in groups.values():
        labels = {row.y for row in group_rows}
        if len(labels) > 1:
            for row in group_rows:
                row.flags.append(conflict_flag)
                if key_attr == "exact":
                    row.tdc_exact_group_row_ids = "|".join(r.row_id for r in group_rows)
                if key_attr == "salt_removed":
                    row.tdc_salt_group_row_ids = "|".join(r.row_id for r in group_rows)
        elif len(group_rows) > 1:
            for row in group_rows:
                row.flags.append(duplicate_flag)
                if key_attr == "exact":
                    row.tdc_exact_group_row_ids = "|".join(r.row_id for r in group_rows)
                if key_attr == "salt_removed":
                    row.tdc_salt_group_row_ids = "|".join(r.row_id for r in group_rows)
    return groups


def apply_b3db_flags(rows: list[Row], b3db_meta: dict[str, Any]) -> None:
    exact_map = b3db_meta["exact_map"]
    salt_map = b3db_meta["salt_map"]

    for row in rows:
        if not row.canon.valid:
            row.flags.append("invalid_rdkit_smiles")
            continue

        exact_entry = exact_map.get(row.canon.exact)
        salt_entry = salt_map.get(row.canon.salt_removed)
        exact_labels = set(exact_entry["labels"]) if exact_entry else set()
        salt_labels = set(salt_entry["labels"]) if salt_entry else set()
        row.b3db_exact_labels = labels_as_text(exact_labels)
        row.b3db_salt_labels = labels_as_text(salt_labels)
        if exact_entry:
            row.b3db_exact_row_nos = "|".join(exact_entry["row_nos"])
            row.b3db_exact_compound_names = " ; ".join(exact_entry["compound_names"])
            row.b3db_exact_smiles = " ; ".join(exact_entry["smiles"])
        if salt_entry:
            row.b3db_salt_row_nos = "|".join(salt_entry["row_nos"])
            row.b3db_salt_compound_names = " ; ".join(salt_entry["compound_names"])
            row.b3db_salt_smiles = " ; ".join(salt_entry["smiles"])

        if len(exact_labels) > 1:
            row.flags.append("b3db_exact_label_ambiguous")
        if len(salt_labels) > 1:
            row.flags.append("b3db_salt_removed_label_ambiguous")

        if len(exact_labels) == 1:
            row.b3db_match_type = "exact"
            if row.y not in exact_labels:
                row.flags.append("b3db_exact_label_mismatch")

        if len(salt_labels) == 1:
            if row.b3db_match_type == "unmatched":
                row.b3db_match_type = "salt_removed"
            if row.y not in salt_labels:
                row.flags.append("b3db_salt_removed_label_mismatch")

        if not exact_labels and not salt_labels:
            row.b3db_match_type = "unmatched"

        if row.canon.salt_removed_changed:
            row.flags.append("salt_removed_changed")


def row_has_hard_problem(row: Row) -> bool:
    hard_flags = {
        "invalid_rdkit_smiles",
        "tdc_exact_label_conflict",
        "tdc_salt_removed_label_conflict",
        "b3db_exact_label_mismatch",
        "b3db_salt_removed_label_mismatch",
    }
    return any(flag in hard_flags for flag in row.flags)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_to_audit_dict(row: Row) -> dict[str, Any]:
    return {
        "source": row.source,
        "split": row.split,
        "row_index": row.row_index,
        "row_id": row.row_id,
        "smiles": row.smiles,
        "Y": row.y,
        "rdkit_valid": row.canon.valid,
        "rdkit_error": row.canon.error,
        "canonical_exact": row.canon.exact,
        "canonical_salt_removed": row.canon.salt_removed,
        "n_fragments": row.canon.n_fragments,
        "b3db_match_type": row.b3db_match_type,
        "b3db_exact_labels": row.b3db_exact_labels,
        "b3db_salt_removed_labels": row.b3db_salt_labels,
        "b3db_exact_row_nos": row.b3db_exact_row_nos,
        "b3db_exact_compound_names": row.b3db_exact_compound_names,
        "b3db_exact_smiles": row.b3db_exact_smiles,
        "b3db_salt_removed_row_nos": row.b3db_salt_row_nos,
        "b3db_salt_removed_compound_names": row.b3db_salt_compound_names,
        "b3db_salt_removed_smiles": row.b3db_salt_smiles,
        "tdc_exact_group_row_ids": row.tdc_exact_group_row_ids,
        "tdc_salt_removed_group_row_ids": row.tdc_salt_group_row_ids,
        "flags": "|".join(sorted(set(row.flags))),
    }


def summarize_groups(groups: dict[str, list[Row]], key_name: str) -> list[dict[str, Any]]:
    out = []
    for key, group_rows in groups.items():
        labels = sorted({row.y for row in group_rows})
        original_smiles = sorted({row.smiles for row in group_rows})
        if len(group_rows) <= 1:
            continue
        out.append(
            {
                key_name: key,
                "num_rows": len(group_rows),
                "num_unique_input_smiles": len(original_smiles),
                "labels": "|".join(str(v) for v in labels),
                "splits": "|".join(sorted({row.split for row in group_rows})),
                "row_ids": "|".join(row.row_id for row in group_rows),
                "input_smiles": " ; ".join(original_smiles),
            }
        )
    out.sort(key=lambda x: (-int(x["num_rows"]), x[key_name]))
    return out


def summarize_b3db_conflicts(label_map: dict[str, dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    out = []
    for key, entry in label_map.items():
        if len(entry["labels"]) <= 1:
            continue
        out.append(
            {
                key_name: key,
                "labels": labels_as_text(entry["labels"]),
                "b3db_row_nos": "|".join(entry["row_nos"]),
                "compound_names": " ; ".join(entry["compound_names"]),
                "smiles": " ; ".join(entry["smiles"]),
            }
        )
    out.sort(key=lambda x: x[key_name])
    return out


def split_counts(rows: list[Row]) -> dict[str, int]:
    counts = Counter(row.split for row in rows)
    return {split: counts.get(split, 0) for split in SPLITS}


def write_filtered_outputs(rows: list[Row], out_dir: Path, confirmed_only: bool = False) -> tuple[list[Row], list[Row]]:
    kept: list[Row] = []
    removed_duplicates: list[Row] = []
    seen_salt_keys: set[str] = set()

    for row in rows:
        if row_has_hard_problem(row):
            continue
        if confirmed_only and row.b3db_match_type not in {"exact", "salt_removed"}:
            continue
        key = row.canon.salt_removed
        if key in seen_salt_keys:
            row.flags.append("removed_duplicate_in_filtered_export")
            removed_duplicates.append(row)
            continue
        seen_salt_keys.add(key)
        kept.append(row)

    prompt_dir = out_dir / "prompts_salt_removed"
    drug_dir = out_dir / "drug_salt_removed"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    drug_dir.mkdir(parents=True, exist_ok=True)

    kept_by_split: dict[str, list[Row]] = defaultdict(list)
    for row in kept:
        kept_by_split[row.split].append(row)

    for split in SPLITS:
        with (drug_dir / f"{split}.jsonl").open("w") as f_drug, (prompt_dir / f"{split}.jsonl").open("w") as f_prompt:
            for row in kept_by_split.get(split, []):
                f_drug.write(json.dumps({"drug": row.canon.salt_removed, "Y": row.y}, ensure_ascii=False) + "\n")
                prompt_obj = dict(row.raw)
                prompt_obj["text"] = replace_prompt_smiles(prompt_obj["text"], row.canon.salt_removed)
                f_prompt.write(json.dumps(prompt_obj, ensure_ascii=False) + "\n")

    return kept, removed_duplicates


def make_report(
    path: Path,
    b3db_paths: list[Path],
    b3db_meta: dict[str, Any],
    original_rows: list[Row],
    salt_rows: list[Row],
    exact_groups: dict[str, list[Row]],
    salt_groups: dict[str, list[Row]],
    clean_rows: list[Row],
    confirmed_rows: list[Row],
) -> None:
    problem_rows = [row for row in original_rows if row_has_hard_problem(row)]
    mismatch_exact = [row for row in original_rows if "b3db_exact_label_mismatch" in row.flags]
    mismatch_salt = [row for row in original_rows if "b3db_salt_removed_label_mismatch" in row.flags]
    invalid = [row for row in original_rows if "invalid_rdkit_smiles" in row.flags]
    exact_matched = [row for row in original_rows if row.b3db_match_type == "exact"]
    salt_only_matched = [row for row in original_rows if row.b3db_match_type == "salt_removed"]
    unmatched = [row for row in original_rows if row.b3db_match_type == "unmatched"]

    exact_conflict_keys = [k for k, v in exact_groups.items() if len({row.y for row in v}) > 1]
    salt_conflict_keys = [k for k, v in salt_groups.items() if len({row.y for row in v}) > 1]
    salt_collapse = [
        (k, v)
        for k, v in salt_groups.items()
        if len(v) > 1 and len({row.canon.exact for row in v}) > 1
    ]

    lines = [
        "# BBB_Martins vs B3DB Quality Audit",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- B3DB source files: {', '.join(f'`{p}`' for p in b3db_paths if p.exists())}",
        "- B3DB label mapping: `BBB+ -> 1` (crosses BBB), `BBB- -> 0` (does not cross BBB).",
        "- Matching policy: RDKit canonical isomeric SMILES exact match first; RDKit LargestFragmentChooser salt-removed canonical match second.",
        "",
        "## Top-line counts",
        "",
        f"- TDC original BBB_Martins rows: {len(original_rows)}; split counts: {split_counts(original_rows)}",
        f"- TDC no-conflict salt-removed BBB_Martins rows already present: {len(salt_rows)}; split counts: {split_counts(salt_rows)}",
        f"- B3DB classification rows loaded: {b3db_meta['n_rows']} ({b3db_meta['n_valid_rdkit']} RDKit-valid, {b3db_meta['n_invalid_rdkit']} invalid)",
        f"- Original TDC rows with exact B3DB match: {len(exact_matched)}",
        f"- Original TDC rows with salt-removed-only B3DB match: {len(salt_only_matched)}",
        f"- Original TDC rows unmatched to B3DB: {len(unmatched)}",
        f"- Hard-problem original TDC rows removed by the clean filter: {len(problem_rows)}",
        f"- Clean filtered rows after hard-problem removal and salt-removed dedup: {len(clean_rows)}; split counts: {split_counts(clean_rows)}",
        f"- B3DB-confirmed clean rows after the same dedup: {len(confirmed_rows)}; split counts: {split_counts(confirmed_rows)}",
        "",
        "## Label quality against B3DB",
        "",
        f"- Exact-match label mismatches: {len(mismatch_exact)} rows.",
        f"- Salt-removed-match label mismatches: {len(mismatch_salt)} rows.",
        f"- TDC rows whose label is confirmed correct by B3DB exact match: {len([r for r in exact_matched if not row_has_hard_problem(r)])}.",
        f"- TDC rows whose label is confirmed correct only after salt removal: {len([r for r in salt_only_matched if not row_has_hard_problem(r)])}.",
        "",
        "## Internal TDC issues",
        "",
        f"- RDKit-unparseable original TDC rows: {len(invalid)}.",
        f"- Same original canonical molecule with conflicting labels: {len(exact_conflict_keys)} canonical keys.",
        f"- Same salt-removed canonical molecule with conflicting labels: {len(salt_conflict_keys)} canonical keys.",
        f"- Salt removal collapses multiple original molecules into the same canonical molecule: {len(salt_collapse)} salt-removed keys.",
        f"- B3DB exact canonical keys with conflicting labels: {b3db_meta['exact_conflict_keys']}.",
        f"- B3DB salt-removed canonical keys with conflicting labels: {b3db_meta['salt_removed_conflict_keys']}.",
        "",
        "## Output files",
        "",
        "- `reports/problem_rows.csv`: all original TDC rows with hard-removal flags.",
        "- `reports/tdc_b3db_label_mismatches.csv`: rows whose TDC label disagrees with B3DB.",
        "- `reports/tdc_duplicate_groups_exact.csv`: duplicate/conflicting groups by original canonical SMILES.",
        "- `reports/tdc_duplicate_groups_salt_removed.csv`: duplicate/conflicting groups by salt-removed canonical SMILES.",
        "- `reports/b3db_conflict_groups_exact.csv` and `reports/b3db_conflict_groups_salt_removed.csv`: B3DB-internal conflicting canonical keys.",
        "- `processed/BBB_Martins_filtered_clean/drug_salt_removed/{train,valid,test}.jsonl`: clean compatible `drug`/`Y` JSONL.",
        "- `processed/BBB_Martins_filtered_clean/prompts_salt_removed/{train,valid,test}.jsonl`: clean prompt JSONL with SMILES replaced by salt-removed canonical SMILES.",
        "- `processed/BBB_Martins_b3db_confirmed_clean/...`: stricter subset containing only B3DB-matched agreeing molecules.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3db", type=Path, default=Path("DataPrepare/B3DB/raw/B3DB_classification.tsv"))
    parser.add_argument(
        "--b3db-extra",
        type=Path,
        default=Path("DataPrepare/B3DB/raw/B3DB_classification_external.tsv"),
        help="Optional extra B3DB classification TSV, e.g. the 2025 external set.",
    )
    parser.add_argument("--tdc-original", type=Path, default=Path("DataPrepare/TDC_train_prompts_label_scaffold"))
    parser.add_argument("--tdc-salt-removed", type=Path, default=Path("DataPrepare/TDC_no_conflict_labels_salt_removed"))
    parser.add_argument("--out-dir", type=Path, default=Path("DataPrepare/B3DB"))
    args = parser.parse_args()

    b3db_paths = [args.b3db]
    if args.b3db_extra and args.b3db_extra.exists():
        b3db_paths.append(args.b3db_extra)
    _, b3db_meta = load_b3db(b3db_paths)
    original_rows = load_prompt_rows(args.tdc_original)
    salt_rows = load_salt_removed_rows(args.tdc_salt_removed)

    exact_groups = add_group_flags(
        original_rows,
        "exact",
        "tdc_exact_label_conflict",
        "tdc_exact_duplicate_same_label",
    )
    salt_groups = add_group_flags(
        original_rows,
        "salt_removed",
        "tdc_salt_removed_label_conflict",
        "tdc_salt_removed_duplicate_same_label",
    )
    apply_b3db_flags(original_rows, b3db_meta)

    add_group_flags(
        salt_rows,
        "exact",
        "tdc_exact_label_conflict",
        "tdc_exact_duplicate_same_label",
    )
    add_group_flags(
        salt_rows,
        "salt_removed",
        "tdc_salt_removed_label_conflict",
        "tdc_salt_removed_duplicate_same_label",
    )
    apply_b3db_flags(salt_rows, b3db_meta)

    reports_dir = args.out_dir / "reports"
    processed_dir = args.out_dir / "processed"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    audit_fields = [
        "source",
        "split",
        "row_index",
        "row_id",
        "smiles",
        "Y",
        "rdkit_valid",
        "rdkit_error",
        "canonical_exact",
        "canonical_salt_removed",
        "n_fragments",
        "b3db_match_type",
        "b3db_exact_labels",
        "b3db_salt_removed_labels",
        "b3db_exact_row_nos",
        "b3db_exact_compound_names",
        "b3db_exact_smiles",
        "b3db_salt_removed_row_nos",
        "b3db_salt_removed_compound_names",
        "b3db_salt_removed_smiles",
        "tdc_exact_group_row_ids",
        "tdc_salt_removed_group_row_ids",
        "flags",
    ]
    write_csv(reports_dir / "tdc_original_all_rows_audit.csv", [row_to_audit_dict(r) for r in original_rows], audit_fields)
    write_csv(reports_dir / "tdc_salt_removed_all_rows_audit.csv", [row_to_audit_dict(r) for r in salt_rows], audit_fields)
    write_csv(
        reports_dir / "problem_rows.csv",
        [row_to_audit_dict(r) for r in original_rows if row_has_hard_problem(r)],
        audit_fields,
    )
    write_csv(
        reports_dir / "tdc_b3db_label_mismatches.csv",
        [
            row_to_audit_dict(r)
            for r in original_rows
            if "b3db_exact_label_mismatch" in r.flags or "b3db_salt_removed_label_mismatch" in r.flags
        ],
        audit_fields,
    )
    write_csv(
        reports_dir / "tdc_invalid_smiles.csv",
        [row_to_audit_dict(r) for r in original_rows if "invalid_rdkit_smiles" in r.flags],
        audit_fields,
    )

    group_fields_exact = [
        "canonical_exact",
        "num_rows",
        "num_unique_input_smiles",
        "labels",
        "splits",
        "row_ids",
        "input_smiles",
    ]
    group_fields_salt = [
        "canonical_salt_removed",
        "num_rows",
        "num_unique_input_smiles",
        "labels",
        "splits",
        "row_ids",
        "input_smiles",
    ]
    exact_group_rows = summarize_groups(exact_groups, "canonical_exact")
    salt_group_rows = summarize_groups(salt_groups, "canonical_salt_removed")
    write_csv(reports_dir / "tdc_duplicate_groups_exact.csv", exact_group_rows, group_fields_exact)
    write_csv(reports_dir / "tdc_duplicate_groups_salt_removed.csv", salt_group_rows, group_fields_salt)
    write_csv(
        reports_dir / "salt_collapse_groups.csv",
        [r for r in salt_group_rows if int(r["num_unique_input_smiles"]) > 1],
        group_fields_salt,
    )
    b3db_conflict_fields_exact = ["canonical_exact", "labels", "b3db_row_nos", "compound_names", "smiles"]
    b3db_conflict_fields_salt = ["canonical_salt_removed", "labels", "b3db_row_nos", "compound_names", "smiles"]
    write_csv(
        reports_dir / "b3db_conflict_groups_exact.csv",
        summarize_b3db_conflicts(b3db_meta["exact_map"], "canonical_exact"),
        b3db_conflict_fields_exact,
    )
    write_csv(
        reports_dir / "b3db_conflict_groups_salt_removed.csv",
        summarize_b3db_conflicts(b3db_meta["salt_map"], "canonical_salt_removed"),
        b3db_conflict_fields_salt,
    )

    clean_rows, clean_dup_removed = write_filtered_outputs(
        original_rows,
        processed_dir / "BBB_Martins_filtered_clean",
        confirmed_only=False,
    )
    confirmed_rows, confirmed_dup_removed = write_filtered_outputs(
        original_rows,
        processed_dir / "BBB_Martins_b3db_confirmed_clean",
        confirmed_only=True,
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "b3db_paths": [str(p) for p in b3db_paths if p.exists()],
        "b3db_rows": b3db_meta["n_rows"],
        "b3db_valid_rdkit_rows": b3db_meta["n_valid_rdkit"],
        "b3db_invalid_rdkit_rows": b3db_meta["n_invalid_rdkit"],
        "b3db_exact_conflict_keys": b3db_meta["exact_conflict_keys"],
        "b3db_salt_removed_conflict_keys": b3db_meta["salt_removed_conflict_keys"],
        "tdc_original_rows": len(original_rows),
        "tdc_original_split_counts": split_counts(original_rows),
        "tdc_salt_removed_existing_rows": len(salt_rows),
        "tdc_salt_removed_existing_split_counts": split_counts(salt_rows),
        "tdc_original_invalid_rdkit_rows": sum("invalid_rdkit_smiles" in r.flags for r in original_rows),
        "tdc_exact_label_conflict_rows": sum("tdc_exact_label_conflict" in r.flags for r in original_rows),
        "tdc_salt_removed_label_conflict_rows": sum("tdc_salt_removed_label_conflict" in r.flags for r in original_rows),
        "tdc_exact_duplicate_same_label_rows": sum("tdc_exact_duplicate_same_label" in r.flags for r in original_rows),
        "tdc_salt_removed_duplicate_same_label_rows": sum("tdc_salt_removed_duplicate_same_label" in r.flags for r in original_rows),
        "tdc_b3db_exact_label_mismatch_rows": sum("b3db_exact_label_mismatch" in r.flags for r in original_rows),
        "tdc_b3db_salt_removed_label_mismatch_rows": sum("b3db_salt_removed_label_mismatch" in r.flags for r in original_rows),
        "tdc_b3db_exact_matched_rows": sum(r.b3db_match_type == "exact" for r in original_rows),
        "tdc_b3db_salt_removed_only_matched_rows": sum(r.b3db_match_type == "salt_removed" for r in original_rows),
        "tdc_b3db_unmatched_rows": sum(r.b3db_match_type == "unmatched" for r in original_rows),
        "hard_problem_rows": sum(row_has_hard_problem(r) for r in original_rows),
        "filtered_clean_rows": len(clean_rows),
        "filtered_clean_split_counts": split_counts(clean_rows),
        "filtered_clean_duplicate_rows_removed": len(clean_dup_removed),
        "b3db_confirmed_clean_rows": len(confirmed_rows),
        "b3db_confirmed_clean_split_counts": split_counts(confirmed_rows),
        "b3db_confirmed_duplicate_rows_removed": len(confirmed_dup_removed),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    make_report(
        reports_dir / "BBB_Martins_quality_report.md",
        b3db_paths,
        b3db_meta,
        original_rows,
        salt_rows,
        exact_groups,
        salt_groups,
        clean_rows,
        confirmed_rows,
    )


if __name__ == "__main__":
    main()
