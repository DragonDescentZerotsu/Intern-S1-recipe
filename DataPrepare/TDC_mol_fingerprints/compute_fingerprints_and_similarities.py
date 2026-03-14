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


def process_task(task_name, group_name):
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
         
    # Scaffold split (70/10/20 train/valid/test default)
    # As per user request, just process train and valid.
    split = data.get_split(method='scaffold')
    
    train_df = split['train'].copy()
    valid_df = split['valid'].copy()
    
    # Remove salts before anything else
    logger.info("Removing salts from train set SMILES...")
    train_df['Drug'] = [remove_salts(sm) for sm in train_df['Drug']]
    
    logger.info("Removing salts from valid set SMILES...")
    valid_df['Drug'] = [remove_salts(sm) for sm in valid_df['Drug']]
    
    train_smiles_unique = train_df['Drug'].unique().tolist()
    valid_smiles_unique = valid_df['Drug'].unique().tolist()
    
    logger.info(f"{task_name} - Train SMILES: {len(train_smiles_unique)}, Valid SMILES: {len(valid_smiles_unique)}")
    
    # 2. Compute FPs
    # Use generic mp.Pool
    num_cpus = max(1, mp.cpu_count() - 2)
    logger.info(f"Using {num_cpus} CPUs for parallelization...")
    
    morgan_train = {}
    morgan_valid = {}
    feat_morgan_train = {}
    feat_morgan_valid = {}
    
    with mp.Pool(num_cpus) as pool:
        logger.info(f"Computing Morgan FP for {task_name} train set...")
        results = list(tqdm(pool.imap(partial(compute_single_fingerprint, use_features=False), train_smiles_unique), total=len(train_smiles_unique)))
        for sm, fp in results:
            if fp is not None: morgan_train[sm] = fp
            
        logger.info(f"Computing Feature Morgan FP for {task_name} train set...")
        results = list(tqdm(pool.imap(partial(compute_single_fingerprint, use_features=True), train_smiles_unique), total=len(train_smiles_unique)))
        for sm, fp in results:
            if fp is not None: feat_morgan_train[sm] = fp
            
        logger.info(f"Computing Morgan FP for {task_name} valid set...")
        results = list(tqdm(pool.imap(partial(compute_single_fingerprint, use_features=False), valid_smiles_unique), total=len(valid_smiles_unique)))
        for sm, fp in results:
            if fp is not None: morgan_valid[sm] = fp
            
        logger.info(f"Computing Feature Morgan FP for {task_name} valid set...")
        results = list(tqdm(pool.imap(partial(compute_single_fingerprint, use_features=True), valid_smiles_unique), total=len(valid_smiles_unique)))
        for sm, fp in results:
            if fp is not None: feat_morgan_valid[sm] = fp
            
    # 3. Save FPs
    with open(morgan_dir / "train.pkl", "wb") as f:
        pickle.dump(morgan_train, f)
    with open(morgan_dir / "valid.pkl", "wb") as f:
        pickle.dump(morgan_valid, f)
    with open(feature_morgan_dir / "train.pkl", "wb") as f:
        pickle.dump(feat_morgan_train, f)
    with open(feature_morgan_dir / "valid.pkl", "wb") as f:
        pickle.dump(feat_morgan_valid, f)
        
    logger.info(f"{task_name} - Fingerprints successfully saved.")

    safe_labels_by_split = {}

    # Save a lookup table that keeps the canonicalized SMILES aligned with labels.
    for split_name, df in (("train", train_df), ("valid", valid_df)):
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
    train_labels_map = safe_labels_by_split["train"]
    valid_labels_map = safe_labels_by_split["valid"]
    
    # Convert FP dicts to lists for mapping: (smiles, fp, label)
    morgan_train_list = [(sm, fp, train_labels_map[sm]) for sm, fp in morgan_train.items() if sm in train_labels_map]
    feat_morgan_train_list = [(sm, fp, train_labels_map[sm]) for sm, fp in feat_morgan_train.items() if sm in train_labels_map]
    
    morgan_valid_list = [(sm, fp, valid_labels_map[sm]) for sm, fp in morgan_valid.items() if sm in valid_labels_map]
    feat_morgan_valid_list = [(sm, fp, valid_labels_map[sm]) for sm, fp in feat_morgan_valid.items() if sm in valid_labels_map]
    
    morgan_valid_sims = {}
    morgan_train_sims = {}
    feat_morgan_valid_sims = {}
    feat_morgan_train_sims = {}

    with mp.Pool(num_cpus) as pool:
        # A. Valid vs. Train (Morgan)
        logger.info(f"Computing Valid->Train Morgan Similarity for {task_name}...")
        results = list(tqdm(pool.imap(partial(compute_similarities_for_query, references=morgan_train_list, exclude_self=False), morgan_valid_list), total=len(morgan_valid_list)))
        for sm, sims in results:
            morgan_valid_sims[sm] = sims
            
        # B. Train vs. Train (Morgan)
        logger.info(f"Computing Train->Train Morgan Similarity for {task_name}...")
        results = list(tqdm(pool.imap(partial(compute_similarities_for_query, references=morgan_train_list, exclude_self=True), morgan_train_list), total=len(morgan_train_list)))
        for sm, sims in results:
            morgan_train_sims[sm] = sims

        # C. Valid vs. Train (Feature Morgan)
        logger.info(f"Computing Valid->Train Feature Morgan Similarity for {task_name}...")
        results = list(tqdm(pool.imap(partial(compute_similarities_for_query, references=feat_morgan_train_list, exclude_self=False), feat_morgan_valid_list), total=len(feat_morgan_valid_list)))
        for sm, sims in results:
            feat_morgan_valid_sims[sm] = sims
            
        # D. Train vs. Train (Feature Morgan)
        logger.info(f"Computing Train->Train Feature Morgan Similarity for {task_name}...")
        results = list(tqdm(pool.imap(partial(compute_similarities_for_query, references=feat_morgan_train_list, exclude_self=True), feat_morgan_train_list), total=len(feat_morgan_train_list)))
        for sm, sims in results:
            feat_morgan_train_sims[sm] = sims
            
    # 5. Save Similarities
    with open(morgan_sim_dir / "valid_similarity.pkl", "wb") as f:
        pickle.dump(morgan_valid_sims, f)
    with open(morgan_sim_dir / "train_similarity.pkl", "wb") as f:
        pickle.dump(morgan_train_sims, f)
        
    with open(feature_morgan_sim_dir / "valid_similarity.pkl", "wb") as f:
        pickle.dump(feat_morgan_valid_sims, f)
    with open(feature_morgan_sim_dir / "train_similarity.pkl", "wb") as f:
        pickle.dump(feat_morgan_train_sims, f)
        
    logger.info(f"========== Finished task: {task_name} ==========\n")

def main():
    tasks = [
        ("Carcinogens_Lagunin", "Tox"),
        # ("BBB_Martins", "ADME"),
        # ("DILI", "Tox"),
        # ("Pgp_Broccatelli", "ADME"),
        # ("PAMPA_NCATS", "ADME"),
        # ("HIA_Hou", "ADME"),
        # ("Bioavailability_Ma", "ADME"),
        # ("hERG", "Tox"),
        # ("AMES", "Tox"),
        # ("Skin_Reaction", "Tox"),
        # ("ClinTox", "Tox"),
        # ("CYP2C9_Substrate_CarbonMangels", "ADME"),
        # ("CYP2D6_Substrate_CarbonMangels", "ADME"),
        # ("CYP3A4_Substrate_CarbonMangels", "ADME"),
        # ("SARSCoV2_3CLPro_Diamond", "HTS"),
        # ("SARSCoV2_Vitro_Touret", "HTS"),
    ]
    
    for task_name, group_name in tasks:
        process_task(task_name, group_name)

if __name__ == "__main__":
    main()
