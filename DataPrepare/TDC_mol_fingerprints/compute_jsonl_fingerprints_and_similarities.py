import argparse
import json
import logging
import multiprocessing as mp
import pickle
from functools import partial
from pathlib import Path

from tqdm import tqdm

from compute_fingerprints_and_similarities import (
    VALID_SPLITS,
    compute_similarities_for_query,
    compute_single_fingerprint,
    load_pickle_if_exists,
    remove_salts,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_JSONL_DIR = (
    Path(__file__).parents[1]
    / "B3DB"
    / "processed"
    / "BBB_Martins_final_clean_b3db_high_conf_rescued"
    / "drug_b3db_smiles"
)
DEFAULT_TASK_NAME = "BBB_Martins_final_clean_b3db_high_conf_rescued_b3db_smiles"


def read_jsonl_split(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            smiles = row.get("drug", row.get("Drug"))
            label = row.get("Y", row.get("y", row.get("label")))
            if smiles is None or label is None:
                raise ValueError(f"{path}:{line_number} must contain a SMILES field and a label field.")
            records.append(
                {
                    "original_smiles": smiles,
                    "canonical_smiles": remove_salts(smiles),
                    "Y": int(label),
                }
            )
    return records


def analyze_jsonl_records(records):
    original_groups = {}
    canonical_groups = {}

    for record in records:
        original_smiles = record["original_smiles"]
        canonical_smiles = record["canonical_smiles"]
        label = record["Y"]

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
        canonical_entry["examples"].append({"original_smiles": original_smiles, "label": label})
        canonical_entry["num_rows"] += 1

    raw_conflicts = []
    raw_conflict_smiles = set()
    for entry in original_groups.values():
        unique_labels = sorted({int(value) for value in entry["labels"] if value is not None})
        if len(unique_labels) > 1:
            entry["labels"] = unique_labels
            raw_conflicts.append(entry)
            raw_conflict_smiles.add(entry["original_smiles"])

    label_map_records = []
    canonical_conflicts = []
    introduced_conflicts = []

    for entry in canonical_groups.values():
        unique_labels = sorted({int(value) for value in entry["labels"] if value is not None})
        entry["all_labels"] = unique_labels
        del entry["labels"]
        entry["has_label_conflict"] = len(unique_labels) > 1
        entry["has_raw_label_conflict"] = any(
            example["original_smiles"] in raw_conflict_smiles for example in entry["examples"]
        )
        entry["has_canonicalization_conflict"] = (
            entry["has_label_conflict"] and not entry["has_raw_label_conflict"]
        )

        if entry["has_label_conflict"]:
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
        "label_map_records": label_map_records,
    }


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_fingerprints_for_split(smiles_list, use_features, num_cpus):
    fingerprints = {}
    with mp.Pool(num_cpus) as pool:
        results = list(
            tqdm(
                pool.imap(partial(compute_single_fingerprint, use_features=use_features), smiles_list),
                total=len(smiles_list),
            )
        )
    for smiles, fingerprint in results:
        if fingerprint is not None:
            fingerprints[smiles] = fingerprint
    return fingerprints


def process_jsonl_dataset(
    jsonl_dir,
    task_name,
    requested_splits,
    force_recompute_fingerprints=False,
    force_recompute_similarity=False,
    num_cpus=None,
):
    logger.info("========== Starting local JSONL task: %s ==========", task_name)
    base_dir = Path(__file__).parent
    jsonl_dir = Path(jsonl_dir)
    requested_splits = tuple(dict.fromkeys(requested_splits))
    required_splits = set(requested_splits)
    if any(split_name in requested_splits for split_name in ("valid", "test")):
        required_splits.add("train")

    split_records = {}
    safe_labels_by_split = {}
    split_smiles_unique = {}

    label_map_dir = base_dir / "Label_maps" / "by_task" / task_name
    conflict_dir = base_dir / "Label_conflicts" / "by_task" / task_name
    label_map_dir.mkdir(parents=True, exist_ok=True)
    conflict_dir.mkdir(parents=True, exist_ok=True)

    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue
        split_path = jsonl_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        records = read_jsonl_split(split_path)
        split_records[split_name] = records
        analysis = analyze_jsonl_records(records)
        label_records = analysis["label_map_records"]
        safe_labels_by_split[split_name] = {
            record["canonical_smiles"]: record["Y"] for record in label_records
        }
        split_smiles_unique[split_name] = list(dict.fromkeys(record["canonical_smiles"] for record in records))

        write_jsonl(label_map_dir / f"{split_name}_labels.jsonl", label_records)
        write_jsonl(conflict_dir / f"{split_name}_excluded_conflicts.jsonl", analysis["canonical_conflicts"])

        if analysis["raw_conflicts"]:
            logger.warning(
                "%s %s: %s original SMILES have conflicting labels before canonicalization.",
                task_name,
                split_name,
                len(analysis["raw_conflicts"]),
            )
        if analysis["introduced_conflicts"]:
            logger.warning(
                "%s %s: %s canonical SMILES acquire label conflicts only after canonicalization.",
                task_name,
                split_name,
                len(analysis["introduced_conflicts"]),
            )
        if analysis["canonical_conflicts"]:
            logger.warning(
                "%s %s: excluded %s ambiguous canonical SMILES from label maps and similarity computation.",
                task_name,
                split_name,
                len(analysis["canonical_conflicts"]),
            )

        logger.info(
            "%s - %s rows=%s unique_canonical_smiles=%s safe_label_smiles=%s",
            task_name,
            split_name,
            len(records),
            len(split_smiles_unique[split_name]),
            len(safe_labels_by_split[split_name]),
        )

    metadata = {
        "task_name": task_name,
        "source_jsonl_dir": str(jsonl_dir),
        "requested_splits": list(requested_splits),
        "required_splits": sorted(required_splits),
    }
    with open(label_map_dir / "source_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    morgan_dir = base_dir / "Morgan" / "by_task" / task_name
    feature_morgan_dir = base_dir / "Feature_Morgan" / "by_task" / task_name
    morgan_sim_dir = base_dir / "Morgan_similarity" / "by_task" / task_name
    feature_morgan_sim_dir = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name
    for path in (morgan_dir, feature_morgan_dir, morgan_sim_dir, feature_morgan_sim_dir):
        path.mkdir(parents=True, exist_ok=True)

    if num_cpus is None:
        num_cpus = max(1, mp.cpu_count() - 28)
    logger.info("Using %s CPUs for parallelization.", num_cpus)

    morgan_fps_by_split = {}
    feat_morgan_fps_by_split = {}
    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue
        morgan_path = morgan_dir / f"{split_name}.pkl"
        feat_path = feature_morgan_dir / f"{split_name}.pkl"
        cached_morgan = None if force_recompute_fingerprints else load_pickle_if_exists(morgan_path)
        cached_feat = None if force_recompute_fingerprints else load_pickle_if_exists(feat_path)

        if cached_morgan is not None and cached_feat is not None:
            logger.info("%s - Loading cached fingerprints for %s split.", task_name, split_name)
            morgan_fps_by_split[split_name] = cached_morgan
            feat_morgan_fps_by_split[split_name] = cached_feat
            continue

        smiles_list = split_smiles_unique[split_name]
        logger.info("Computing Morgan FP for %s %s set.", task_name, split_name)
        morgan_fps_by_split[split_name] = compute_fingerprints_for_split(
            smiles_list, use_features=False, num_cpus=num_cpus
        )
        logger.info("Computing Feature Morgan FP for %s %s set.", task_name, split_name)
        feat_morgan_fps_by_split[split_name] = compute_fingerprints_for_split(
            smiles_list, use_features=True, num_cpus=num_cpus
        )

        with open(morgan_path, "wb") as f:
            pickle.dump(morgan_fps_by_split[split_name], f)
        with open(feat_path, "wb") as f:
            pickle.dump(feat_morgan_fps_by_split[split_name], f)

    train_labels_map = safe_labels_by_split["train"]
    morgan_train_list = [
        (smiles, fp, train_labels_map[smiles])
        for smiles, fp in morgan_fps_by_split["train"].items()
        if smiles in train_labels_map
    ]
    feat_morgan_train_list = [
        (smiles, fp, train_labels_map[smiles])
        for smiles, fp in feat_morgan_fps_by_split["train"].items()
        if smiles in train_labels_map
    ]

    with mp.Pool(num_cpus) as pool:
        for split_name in requested_splits:
            morgan_sim_path = morgan_sim_dir / f"{split_name}_similarity.pkl"
            feat_sim_path = feature_morgan_sim_dir / f"{split_name}_similarity.pkl"
            if (
                morgan_sim_path.exists()
                and feat_sim_path.exists()
                and not force_recompute_similarity
            ):
                logger.info("%s - Similarity for %s split already exists; skipping.", task_name, split_name)
                continue

            labels_map = safe_labels_by_split[split_name]
            exclude_self = split_name == "train"
            morgan_query_list = [
                (smiles, fp, labels_map[smiles])
                for smiles, fp in morgan_fps_by_split[split_name].items()
                if smiles in labels_map
            ]
            feat_query_list = [
                (smiles, fp, labels_map[smiles])
                for smiles, fp in feat_morgan_fps_by_split[split_name].items()
                if smiles in labels_map
            ]

            logger.info("Computing %s->Train Morgan similarity for %s.", split_name, task_name)
            morgan_results = list(
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
            with open(morgan_sim_path, "wb") as f:
                pickle.dump({smiles: sims for smiles, sims in morgan_results}, f)

            logger.info("Computing %s->Train Feature Morgan similarity for %s.", split_name, task_name)
            feat_results = list(
                tqdm(
                    pool.imap(
                        partial(
                            compute_similarities_for_query,
                            references=feat_morgan_train_list,
                            exclude_self=exclude_self,
                        ),
                        feat_query_list,
                    ),
                    total=len(feat_query_list),
                )
            )
            with open(feat_sim_path, "wb") as f:
                pickle.dump({smiles: sims for smiles, sims in feat_results}, f)

    logger.info("========== Finished local JSONL task: %s ==========\n", task_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute TDC-style Morgan fingerprints and train-based similarities for local drug/Y JSONL splits."
    )
    parser.add_argument("--jsonl-dir", type=Path, default=DEFAULT_JSONL_DIR)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--num-cpus", type=int, default=None)
    parser.add_argument("--force-recompute-fingerprints", action="store_true")
    parser.add_argument("--force-recompute-similarity", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    process_jsonl_dataset(
        jsonl_dir=args.jsonl_dir,
        task_name=args.task_name,
        requested_splits=args.splits,
        force_recompute_fingerprints=args.force_recompute_fingerprints,
        force_recompute_similarity=args.force_recompute_similarity,
        num_cpus=args.num_cpus,
    )


if __name__ == "__main__":
    main()
