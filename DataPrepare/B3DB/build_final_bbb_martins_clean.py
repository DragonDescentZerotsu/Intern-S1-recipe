#!/usr/bin/env python3
"""Build the final BBB_Martins clean set with strict B3DB plus rescued matches."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


RDLogger.DisableLog("rdApp.*")

SPLITS = ("train", "valid", "test")
HIGH_CONF_GROUPS = {"A", "B"}
PROMPT_RE = re.compile(r"Drug SMILES:\s*(?:<SMILES>)?([^\n<]+)(?:</SMILES>)?")


@dataclass(frozen=True)
class B3DBChoice:
    smiles: str
    y: int
    group: str
    row_nos: str
    compound_names: str
    match_key: str


def canonicalize(smiles: str, *, isomeric: bool = True, salt_removed: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if salt_removed:
        mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        if mol is None:
            return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


def inchikey(smiles: str, *, salt_removed: bool = False) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    if salt_removed:
        mol = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)
        if mol is None:
            return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def label_to_int(label: Any) -> int:
    text = str(label).strip()
    if text == "BBB+":
        return 1
    if text == "BBB-":
        return 0
    raise ValueError(f"Unexpected B3DB label: {label!r}")


def extract_prompt_smiles(text: str) -> str:
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


def load_original_prompts() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        path = Path(f"DataPrepare/TDC_{split}_prompts_label_scaffold/BBB_Martins.jsonl")
        with path.open() as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                row_id = f"{split}:{idx}"
                rows[row_id] = {
                    "row_id": row_id,
                    "split": split,
                    "row_index": idx,
                    "raw": obj,
                    "original_smiles": extract_prompt_smiles(obj["text"]),
                    "original_y": int(obj["Y"]),
                }
    return rows


def load_b3db() -> pd.DataFrame:
    paths = [
        Path("DataPrepare/B3DB/raw/B3DB_classification.tsv"),
        Path("DataPrepare/B3DB/raw/B3DB_classification_external.tsv"),
    ]
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep="\t")
        df.columns = [str(c).lstrip("\ufeff") for c in df.columns]
        df["source_file"] = path.stem
        frames.append(df)
    b3 = pd.concat(frames, ignore_index=True)
    b3["Y"] = b3["BBB+/BBB-"].map(label_to_int)
    b3["group"] = b3["group"].astype(str)
    b3["row_no"] = b3["source_file"] + ":" + b3["NO."].astype(str)
    b3["canonical_exact"] = b3["SMILES"].map(lambda s: canonicalize(s, isomeric=True, salt_removed=False))
    b3["canonical_salt_removed"] = b3["SMILES"].map(lambda s: canonicalize(s, isomeric=True, salt_removed=True))
    b3["noiso"] = b3["SMILES"].map(lambda s: canonicalize(s, isomeric=False, salt_removed=False))
    b3["noiso_salt"] = b3["SMILES"].map(lambda s: canonicalize(s, isomeric=False, salt_removed=True))
    b3["inchikey"] = b3["SMILES"].map(lambda s: inchikey(s, salt_removed=False))
    b3["inchikey_salt"] = b3["SMILES"].map(lambda s: inchikey(s, salt_removed=True))
    return b3


def build_choice_map(b3: pd.DataFrame, key_col: str, high_conf_only: bool) -> dict[str, B3DBChoice]:
    use = b3.dropna(subset=[key_col]).copy()
    if high_conf_only:
        use = use[use["group"].isin(HIGH_CONF_GROUPS)].copy()

    out: dict[str, B3DBChoice] = {}
    group_rank = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    for key, group in use.groupby(key_col, sort=False):
        labels = set(group["Y"].astype(int))
        if len(labels) != 1:
            continue
        group = group.copy()
        group["_group_rank"] = group["group"].map(group_rank).fillna(99)
        group = group.sort_values(by=["_group_rank", "source_file", "NO."])
        first = group.iloc[0]
        out[str(key)] = B3DBChoice(
            smiles=str(first["SMILES"]),
            y=int(first["Y"]),
            group=str(first["group"]),
            row_nos="|".join(group["row_no"].map(str)),
            compound_names=" ; ".join(group["compound_name"].map(str)),
            match_key=str(key),
        )
    return out


def read_existing_confirmed_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = Path("DataPrepare/B3DB/processed/BBB_Martins_b3db_confirmed_clean")
    for split in SPLITS:
        drug_path = root / "drug_salt_removed" / f"{split}.jsonl"
        prompt_path = root / "prompts_salt_removed" / f"{split}.jsonl"
        with drug_path.open() as f_drug, prompt_path.open() as f_prompt:
            for idx, (drug_line, prompt_line) in enumerate(zip(f_drug, f_prompt)):
                drug_obj = json.loads(drug_line)
                prompt_obj = json.loads(prompt_line)
                rows.append(
                    {
                        "source": "strict_b3db_confirmed_clean",
                        "split": split,
                        "row_id": f"existing:{split}:{idx}",
                        "drug": drug_obj["drug"],
                        "Y": int(drug_obj["Y"]),
                        "prompt": prompt_obj,
                        "b3db_group": "",
                        "b3db_row_nos": "",
                        "original_smiles": extract_prompt_smiles(prompt_obj["text"]),
                    }
                )
    return rows


def make_rescued_rows(original_rows: dict[str, dict[str, Any]], b3: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    noiso_map = build_choice_map(b3, "noiso", high_conf_only=True)
    noiso_salt_map = build_choice_map(b3, "noiso_salt", high_conf_only=True)
    ikey_map = build_choice_map(b3, "inchikey", high_conf_only=True)
    ikey_salt_map = build_choice_map(b3, "inchikey_salt", high_conf_only=True)

    relaxed = pd.read_csv("DataPrepare/B3DB/reports/tdc_unmatched_relaxed_nonisomeric_check.csv")
    ikey = pd.read_csv("DataPrepare/B3DB/reports/tdc_u144_inchikey_check.csv")

    candidates: list[dict[str, Any]] = []

    for _, row in relaxed.iterrows():
        if row["relaxed_match_type"] not in {"nonisomeric_exact", "nonisomeric_salt"}:
            continue
        choice = noiso_map.get(str(row["noiso"]))
        match_mode = "nonisomeric_exact_high_conf"
        if choice is None:
            choice = noiso_salt_map.get(str(row["noiso_salt"]))
            match_mode = "nonisomeric_salt_high_conf"
        candidates.append(make_candidate(row["row_id"], choice, match_mode, original_rows))

    for _, row in ikey.iterrows():
        if not (bool(row["b3db_inchikey_exact"]) or bool(row["b3db_inchikey_salt"])):
            continue
        choice = ikey_map.get(str(row["inchikey"]))
        match_mode = "inchikey_exact_high_conf"
        if choice is None:
            choice = ikey_salt_map.get(str(row["inchikey_salt"]))
            match_mode = "inchikey_salt_high_conf"
        candidates.append(make_candidate(row["row_id"], choice, match_mode, original_rows))

    candidate_df = pd.DataFrame(candidates)
    kept = [row for row in candidates if row.get("keep_candidate", False)]
    return kept, candidate_df


def make_candidate(row_id: str, choice: B3DBChoice | None, match_mode: str, original_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    orig = original_rows[row_id]
    out = {
        "source": match_mode,
        "split": orig["split"],
        "row_id": row_id,
        "original_smiles": orig["original_smiles"],
        "original_Y": orig["original_y"],
        "keep_candidate": choice is not None,
        "drop_reason": "" if choice is not None else "no_unique_high_conf_b3db_match",
        "b3db_smiles": "",
        "b3db_Y": "",
        "b3db_group": "",
        "b3db_row_nos": "",
        "b3db_compound_names": "",
    }
    if choice is not None:
        out.update(
            {
                "b3db_smiles": choice.smiles,
                "b3db_Y": choice.y,
                "b3db_group": choice.group,
                "b3db_row_nos": choice.row_nos,
                "b3db_compound_names": choice.compound_names,
            }
        )
    return out


def write_outputs(base_rows: list[dict[str, Any]], rescue_rows: list[dict[str, Any]], original_rows: dict[str, dict[str, Any]], out_root: Path) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    drug_dir = out_root / "drug_b3db_smiles"
    prompt_dir = out_root / "prompts_b3db_smiles"
    report_dir = out_root / "reports"
    drug_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    final_rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    duplicate_drops: list[dict[str, Any]] = []

    def add_row(row: dict[str, Any]) -> None:
        key = canonicalize(row["drug"], isomeric=True, salt_removed=True)
        if key is None:
            row = dict(row)
            row["drop_reason"] = "invalid_final_smiles"
            duplicate_drops.append(row)
            return
        if key in seen_keys:
            row = dict(row)
            row["drop_reason"] = "duplicate_final_salt_removed_canonical"
            duplicate_drops.append(row)
            return
        seen_keys.add(key)
        final_rows.append(row)

    for row in base_rows:
        add_row(row)

    for row in rescue_rows:
        orig = original_rows[row["row_id"]]
        prompt_obj = dict(orig["raw"])
        prompt_obj["text"] = replace_prompt_smiles(prompt_obj["text"], str(row["b3db_smiles"]))
        prompt_obj["Y"] = int(row["b3db_Y"])
        add_row(
            {
                "source": row["source"],
                "split": row["split"],
                "row_id": row["row_id"],
                "drug": str(row["b3db_smiles"]),
                "Y": int(row["b3db_Y"]),
                "prompt": prompt_obj,
                "b3db_group": row["b3db_group"],
                "b3db_row_nos": row["b3db_row_nos"],
                "original_smiles": row["original_smiles"],
            }
        )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        by_split[row["split"]].append(row)

    for split in SPLITS:
        with (drug_dir / f"{split}.jsonl").open("w") as f_drug, (prompt_dir / f"{split}.jsonl").open("w") as f_prompt:
            for row in by_split[split]:
                f_drug.write(json.dumps({"drug": row["drug"], "Y": row["Y"]}, ensure_ascii=False) + "\n")
                f_prompt.write(json.dumps(row["prompt"], ensure_ascii=False) + "\n")

    fields = [
        "source",
        "split",
        "row_id",
        "drug",
        "Y",
        "original_smiles",
        "b3db_group",
        "b3db_row_nos",
    ]
    with (report_dir / "final_rows_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    with (report_dir / "duplicate_or_invalid_drops.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields + ["drop_reason"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(duplicate_drops)

    counts = Counter(row["split"] for row in final_rows)
    source_counts = Counter(row["source"] for row in final_rows)
    summary = {
        "original_tdc_bbb_martins_rows": 2030,
        "final_rows": len(final_rows),
        "final_split_counts": {split: counts.get(split, 0) for split in SPLITS},
        "lost_vs_original_rows": 2030 - len(final_rows),
        "loss_fraction_vs_original": (2030 - len(final_rows)) / 2030,
        "source_counts": dict(source_counts),
        "duplicate_or_invalid_drops": len(duplicate_drops),
        "high_conf_groups_for_rescues": sorted(HIGH_CONF_GROUPS),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    original_rows = load_original_prompts()
    b3 = load_b3db()
    base_rows = read_existing_confirmed_rows()
    rescue_rows, candidate_df = make_rescued_rows(original_rows, b3)

    out_root = Path("DataPrepare/B3DB/processed/BBB_Martins_final_clean_b3db_high_conf_rescued")
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(report_dir / "rescue_candidate_audit.csv", index=False)
    pd.DataFrame(rescue_rows).to_csv(report_dir / "rescued_rows_kept_before_dedup.csv", index=False)
    summary = write_outputs(base_rows, rescue_rows, original_rows, out_root)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
