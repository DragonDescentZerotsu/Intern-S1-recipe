import argparse
import os
import json
import logging
import pickle
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from functools import partial

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

# Reset RDKit's internal core usage to prevent threading collisions
rdBase.DisableLog('rdApp.error')
os.environ["RDKIT_MAX_THREADS"] = "1"

# TDC imports
from tdc.single_pred import ADME, HTS, Tox

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VALID_SPLITS = ("train", "valid", "test")


def load_pickle_if_exists(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_single_fingerprint(smiles, use_features=False):
    """
    Compute Morgan fingerprint for a single SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles, None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useFeatures=use_features)
        return smiles, fp
    except Exception as e:
        logger.warning(f"Failed generating generic Morgan fp for {smiles}: {e}")
        return smiles, None


def compute_similarities_for_query(query_tuple, references, exclude_self=False):
    """
    Compare a single query fingerprint against a list of references.
    query_tuple is (query_smiles, query_fp).
    references is a list of tuples: (ref_smiles, ref_fp, ref_label).
    Returns (query_smiles, {"label_0": sorted_sims_0, "label_1": sorted_sims_1}).
    """
    query_smiles, query_fp, query_label = query_tuple
    if query_fp is None:
        return query_smiles, {"query_label": int(query_label), "label_0": [], "label_1": []}
        
    ref_smiles_list = [ref[0] for ref in references]
    ref_fp_list = [ref[1] for ref in references]
    ref_label_list = [ref[2] for ref in references]
    
    # Calculate bulk similarity
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fp_list)
    
    label_0_results = []
    label_1_results = []
    
    for score, ref_smiles, ref_label in zip(similarities, ref_smiles_list, ref_label_list):
        if exclude_self and query_smiles == ref_smiles:
            continue
        if int(ref_label) == 0:
            label_0_results.append((score, ref_smiles))
        else:
            label_1_results.append((score, ref_smiles))
        
    # Sort DESCENDING by score
    label_0_results.sort(key=lambda x: x[0], reverse=True)
    label_1_results.sort(key=lambda x: x[0], reverse=True)
    
    return query_smiles, {"query_label": int(query_label), "label_0": label_0_results, "label_1": label_1_results}


def remove_salts(smiles: str) -> str:
    """
    Remove salts/counterions from a SMILES string by keeping the largest
    organic fragment, then return the canonical SMILES of the desalted molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
        cleaned = lfc.choose(mol)
        if cleaned is not None:
            return Chem.MolToSmiles(cleaned, canonical=True, isomericSmiles=True)
    except Exception:
        pass
    return smiles


def analyze_split_labels(split_df, canonical_df):
    """Group labels by original and canonical SMILES to find ambiguous entries."""
    original_groups = {}
    canonical_groups = {}

    for original_smiles, canonical_smiles, label in zip(split_df["Drug"], canonical_df["Drug"], canonical_df["Y"]):
        original_entry = original_groups.setdefault(
            original_smiles,
            {"original_smiles": original_smiles, "labels": [], "num_rows": 0},
        )
        original_entry["labels"].append(label)
        original_entry["num_rows"] += 1

        canonical_entry = canonical_groups.setdefault(
            canonical_smiles,
            {
                "canonical_smiles": canonical_smiles,
                "labels": [],
                "examples": [],
                "num_rows": 0,
            },
        )
        canonical_entry["labels"].append(label)
        canonical_entry["examples"].append(
            {
                "original_smiles": original_smiles,
                "label": label,
            }
        )
        canonical_entry["num_rows"] += 1

    raw_conflicts = []
    raw_conflict_smiles = set()
    for entry in original_groups.values():
        unique_labels = sorted({int(v) for v in entry["labels"] if v is not None})
        if len(unique_labels) > 1:
            entry["labels"] = unique_labels
            raw_conflicts.append(entry)
            raw_conflict_smiles.add(entry["original_smiles"])

    label_map_records = []
    canonical_conflicts = []
    introduced_conflicts = []
    ambiguous_canonical_smiles = set()

    for entry in canonical_groups.values():
        unique_labels = sorted({int(v) for v in entry["labels"] if v is not None})
        entry["all_labels"] = unique_labels
        del entry["labels"]
        entry["has_label_conflict"] = len(unique_labels) > 1
        entry["has_raw_label_conflict"] = any(
            example["original_smiles"] in raw_conflict_smiles for example in entry["examples"]
        )
        entry["has_canonicalization_conflict"] = entry["has_label_conflict"] and not entry["has_raw_label_conflict"]

        if entry["has_label_conflict"]:
            ambiguous_canonical_smiles.add(entry["canonical_smiles"])
            canonical_conflicts.append(entry)
            if entry["has_canonicalization_conflict"]:
                introduced_conflicts.append(entry)
            continue

        if unique_labels:
            entry["Y"] = unique_labels[0]
            label_map_records.append(entry)

    return {
        "raw_conflicts": raw_conflicts,
        "canonical_conflicts": canonical_conflicts,
        "introduced_conflicts": introduced_conflicts,
        "ambiguous_canonical_smiles": ambiguous_canonical_smiles,
        "label_map_records": label_map_records,
    }


def process_task(task_name, group_name, requested_splits):
    """
    Loads data using TDC, computes fingerprints in parallel, and calculates similarities.
    """
    logger.info(f"========== Starting task: {task_name} ==========")
    
    # Paths setup
    base_dir = Path(__file__).parent
    
    morgan_dir = base_dir / "Morgan" / "by_task" / task_name
    feature_morgan_dir = base_dir / "Feature_Morgan" / "by_task" / task_name
    morgan_sim_dir = base_dir / "Morgan_similarity" / "by_task" / task_name
    feature_morgan_sim_dir = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name
    label_map_dir = base_dir / "Label_maps" / "by_task" / task_name
    conflict_dir = base_dir / "Label_conflicts" / "by_task" / task_name
    
    morgan_dir.mkdir(parents=True, exist_ok=True)
    feature_morgan_dir.mkdir(parents=True, exist_ok=True)
    morgan_sim_dir.mkdir(parents=True, exist_ok=True)
    feature_morgan_sim_dir.mkdir(parents=True, exist_ok=True)
    label_map_dir.mkdir(parents=True, exist_ok=True)
    conflict_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    data = None
    if group_name == 'Tox':
        data = Tox(name=task_name)
    elif group_name == 'ADME':
        data = ADME(name=task_name)
    elif group_name == 'HTS':
        data = HTS(name=task_name)
    else:
         logger.error(f"Unknown group {group_name} for task {task_name}.")
         return
         
    requested_splits = tuple(dict.fromkeys(requested_splits))
    required_splits = set(requested_splits)
    if any(split_name in requested_splits for split_name in ("valid", "test")):
        required_splits.add("train")

    # Scaffold split (70/10/20 train/valid/test default)
    split = data.get_split(method='scaffold')

    split_dfs = {}
    split_smiles_unique = {}

    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue
        split_df = split[split_name].copy()
        logger.info("Removing salts from %s set SMILES...", split_name)
        split_df["Drug"] = [remove_salts(sm) for sm in split_df["Drug"]]
        split_dfs[split_name] = split_df
        split_smiles_unique[split_name] = split_df["Drug"].unique().tolist()
        logger.info(
            "%s - %s SMILES: %s",
            task_name,
            split_name.capitalize(),
            len(split_smiles_unique[split_name]),
        )
    
    # 2. Load or compute FPs
    num_cpus = max(1, mp.cpu_count() - 2)
    logger.info(f"Using {num_cpus} CPUs for parallelization...")
    
    morgan_fps_by_split = {split_name: {} for split_name in required_splits}
    feat_morgan_fps_by_split = {split_name: {} for split_name in required_splits}
    computed_fp_splits = set()

    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue

        morgan_path = morgan_dir / f"{split_name}.pkl"
        feat_morgan_path = feature_morgan_dir / f"{split_name}.pkl"
        cached_morgan = load_pickle_if_exists(morgan_path)
        cached_feat_morgan = load_pickle_if_exists(feat_morgan_path)

        if cached_morgan is not None and cached_feat_morgan is not None:
            logger.info(
                "%s - Loading cached fingerprints for %s split.",
                task_name,
                split_name,
            )
            morgan_fps_by_split[split_name] = cached_morgan
            feat_morgan_fps_by_split[split_name] = cached_feat_morgan
            continue

        computed_fp_splits.add(split_name)

    if computed_fp_splits:
        with mp.Pool(num_cpus) as pool:
            for split_name in VALID_SPLITS:
                if split_name not in computed_fp_splits:
                    continue
                smiles_unique = split_smiles_unique[split_name]

                logger.info(f"Computing Morgan FP for {task_name} {split_name} set...")
                results = list(
                    tqdm(
                        pool.imap(partial(compute_single_fingerprint, use_features=False), smiles_unique),
                        total=len(smiles_unique),
                    )
                )
                for sm, fp in results:
                    if fp is not None:
                        morgan_fps_by_split[split_name][sm] = fp

                logger.info(f"Computing Feature Morgan FP for {task_name} {split_name} set...")
                results = list(
                    tqdm(
                        pool.imap(partial(compute_single_fingerprint, use_features=True), smiles_unique),
                        total=len(smiles_unique),
                    )
                )
                for sm, fp in results:
                    if fp is not None:
                        feat_morgan_fps_by_split[split_name][sm] = fp
            
    # 3. Save FPs
    for split_name in computed_fp_splits:
        with open(morgan_dir / f"{split_name}.pkl", "wb") as f:
            pickle.dump(morgan_fps_by_split[split_name], f)
        with open(feature_morgan_dir / f"{split_name}.pkl", "wb") as f:
            pickle.dump(feat_morgan_fps_by_split[split_name], f)
        
    logger.info(f"{task_name} - Fingerprints are ready.")

    safe_labels_by_split = {}

    # Save a lookup table that keeps the canonicalized SMILES aligned with labels.
    for split_name in required_splits:
        df = split_dfs[split_name]
        analysis = analyze_split_labels(split[split_name], df)
        records = analysis["label_map_records"]
        safe_labels_by_split[split_name] = {
            record["canonical_smiles"]: record["Y"] for record in records
        }

        out_path = label_map_dir / f"{split_name}_labels.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        conflict_records = analysis["canonical_conflicts"]
        conflict_path = conflict_dir / f"{split_name}_excluded_conflicts.jsonl"
        with open(conflict_path, "w", encoding="utf-8") as f:
            for record in conflict_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        raw_conflicts = analysis["raw_conflicts"]
        if raw_conflicts:
            logger.warning(
                f"{task_name} {split_name}: {len(raw_conflicts)} original SMILES already have conflicting labels before canonicalization."
            )
            for record in raw_conflicts:
                logger.warning(
                    "Raw label conflict for original SMILES %s | labels=%s | num_rows=%s",
                    record["original_smiles"],
                    record["labels"],
                    record["num_rows"],
                )

        introduced_conflicts = analysis["introduced_conflicts"]
        if introduced_conflicts:
            logger.warning(
                f"{task_name} {split_name}: {len(introduced_conflicts)} canonical SMILES acquire label conflicts only after canonicalization."
            )
            for record in introduced_conflicts:
                logger.warning(
                    "Canonicalization-introduced conflict for canonical SMILES %s | labels=%s | examples=%s",
                    record["canonical_smiles"],
                    record["all_labels"],
                    record["examples"],
                )

        total_conflicts = len(conflict_records)
        if total_conflicts:
            logger.warning(
                f"{task_name} {split_name}: excluded {total_conflicts} ambiguous canonical SMILES from label maps and similarity computation."
            )
    
    # 4. Compute Similarities
    similarity_targets = [split_name for split_name in requested_splits if split_name in ("train", "valid", "test")]
    similarity_results = {
        "morgan": {},
        "feature_morgan": {},
    }
    
    # Convert FP dicts to lists for mapping: (smiles, fp, label)
    train_labels_map = safe_labels_by_split["train"]
    morgan_train_list = [
        (sm, fp, train_labels_map[sm])
        for sm, fp in morgan_fps_by_split["train"].items()
        if sm in train_labels_map
    ]
    feat_morgan_train_list = [
        (sm, fp, train_labels_map[sm])
        for sm, fp in feat_morgan_fps_by_split["train"].items()
        if sm in train_labels_map
    ]

    with mp.Pool(num_cpus) as pool:
        for split_name in similarity_targets:
            labels_map = safe_labels_by_split[split_name]
            morgan_query_list = [
                (sm, fp, labels_map[sm])
                for sm, fp in morgan_fps_by_split[split_name].items()
                if sm in labels_map
            ]
            feat_morgan_query_list = [
                (sm, fp, labels_map[sm])
                for sm, fp in feat_morgan_fps_by_split[split_name].items()
                if sm in labels_map
            ]
            exclude_self = split_name == "train"

            logger.info(
                "Computing %s->Train Morgan Similarity for %s...",
                split_name.capitalize(),
                task_name,
            )
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            compute_similarities_for_query,
                            references=morgan_train_list,
                            exclude_self=exclude_self,
                        ),
                        morgan_query_list,
                    ),
                    total=len(morgan_query_list),
                )
            )
            similarity_results["morgan"][split_name] = {sm: sims for sm, sims in results}

            logger.info(
                "Computing %s->Train Feature Morgan Similarity for %s...",
                split_name.capitalize(),
                task_name,
            )
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            compute_similarities_for_query,
                            references=feat_morgan_train_list,
                            exclude_self=exclude_self,
                        ),
                        feat_morgan_query_list,
                    ),
                    total=len(feat_morgan_query_list),
                )
            )
            similarity_results["feature_morgan"][split_name] = {sm: sims for sm, sims in results}
            
    # 5. Save Similarities
    for split_name in requested_splits:
        with open(morgan_sim_dir / f"{split_name}_similarity.pkl", "wb") as f:
            pickle.dump(similarity_results["morgan"][split_name], f)

        with open(feature_morgan_sim_dir / f"{split_name}_similarity.pkl", "wb") as f:
            pickle.dump(similarity_results["feature_morgan"][split_name], f)
        
    logger.info(f"========== Finished task: {task_name} ==========\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute fingerprints and train-based similarities for selected TDC splits."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["train", "valid"],
        help="Which splits to output. valid/test similarities are always computed against train.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    tasks = [
        ("Carcinogens_Lagunin", "Tox"),
        ("BBB_Martins", "ADME"),
        ("DILI", "Tox"),
        ("Pgp_Broccatelli", "ADME"),
        ("PAMPA_NCATS", "ADME"),
        ("HIA_Hou", "ADME"),
        ("Bioavailability_Ma", "ADME"),
        ("hERG", "Tox"),
        ("AMES", "Tox"),
        ("Skin_Reaction", "Tox"),
        ("ClinTox", "Tox"),
        ("CYP2C9_Substrate_CarbonMangels", "ADME"),
        ("CYP2D6_Substrate_CarbonMangels", "ADME"),
        ("CYP3A4_Substrate_CarbonMangels", "ADME"),
        ("SARSCoV2_3CLPro_Diamond", "HTS"),
        ("SARSCoV2_Vitro_Touret", "HTS"),
    ]
    
    for task_name, group_name in tasks:
        process_task(task_name, group_name, args.splits)

if __name__ == "__main__":
    main()
