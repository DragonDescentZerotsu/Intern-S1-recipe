import argparse
import json
import logging
import os
import pickle
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from compute_fingerprints_and_similarities import (
    VALID_SPLITS,
    analyze_split_labels,
    export_clean_split_records,
    load_pickle_if_exists,
    remove_salts,
)

from tdc.single_pred import ADME, HTS, Tox


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MINIMOL_SOURCE = Path("/data1/tianang/Projects/minimol")
TASKS = [
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
TASK_GROUPS = dict(TASKS)


def patch_graphium_float16_adjacency():
    """
    Graphium 2.4.7 passes float16 adjacency matrices into scipy.sparse, which
    newer SciPy rejects. MiniMol casts tensors to FP32 anyway, so force only the
    adjacency construction to SciPy-compatible FP32.
    """
    import graphium.features.featurizer as featurizer

    if getattr(featurizer.mol_to_adjacency_matrix, "_minimol_fp32_patch", False):
        return

    original = featurizer.mol_to_adjacency_matrix

    def patched_mol_to_adjacency_matrix(*args, **kwargs):
        if kwargs.get("dtype", np.float32) == np.float16:
            kwargs = dict(kwargs)
            kwargs["dtype"] = np.float32
        return original(*args, **kwargs)

    patched_mol_to_adjacency_matrix._minimol_fp32_patch = True
    featurizer.mol_to_adjacency_matrix = patched_mol_to_adjacency_matrix


def source_has_real_minimol_checkpoint(minimol_source):
    if not minimol_source:
        return False
    state_dict_path = minimol_source / "minimol" / "ckpts" / "minimol_v1" / "state_dict.pth"
    if not state_dict_path.exists():
        return False
    try:
        with open(state_dict_path, "rb") as f:
            prefix = f.read(64)
    except OSError:
        return False
    return not prefix.startswith(b"version https://git-lfs.github.com/spec/")


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_arg}, but torch.cuda.is_available() is False")
    return device


def cast_feature_tensors_to_fp32(input_features):
    for input_feature in input_features:
        if isinstance(input_feature, str):
            continue
        for key, value in input_feature.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    input_feature[key] = value.float()
                elif value.dtype == torch.int32:
                    input_feature[key] = value.long()
    return input_features


class MiniMolEmbedder:
    def __init__(
        self,
        batch_size,
        device,
        featurization_jobs,
        featurization_backend,
        featurization_batch_size,
        minimol_source,
    ):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-minimol")
        if minimol_source and minimol_source.exists() and source_has_real_minimol_checkpoint(minimol_source):
            sys.path.insert(0, str(minimol_source))
            logger.info("Using MiniMol source tree: %s", minimol_source)
        elif minimol_source and minimol_source.exists():
            logger.warning(
                "Skipping MiniMol source tree %s because its state_dict.pth is missing or still a Git LFS pointer; "
                "falling back to the installed minimol package.",
                minimol_source,
            )

        patch_graphium_float16_adjacency()

        from minimol import Minimol
        from torch_geometric.data import Batch
        from torch_geometric.nn import global_max_pool

        self.batch_cls = Batch
        self.global_max_pool = global_max_pool
        self.batch_size = batch_size
        self.device = device

        logger.info("Initializing MiniMol on %s...", device)
        original_torch_load = torch.load

        def torch_load_compat(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_compat
        try:
            self.model = Minimol(batch_size=batch_size)
        finally:
            torch.load = original_torch_load
        self.model.datamodule.featurization_n_jobs = featurization_jobs
        self.model.datamodule.featurization_backend = featurization_backend
        self.model.datamodule.featurization_batch_size = featurization_batch_size
        self.model.datamodule.smiles_transformer.keywords["dtype"] = np.float32

        predictor = self.model.predictor.predictor
        predictor.to(device)
        predictor.eval()
        self.model.predictor.network.eval()
        logger.info("MiniMol parameter device: %s", next(predictor.parameters()).device)

    def embed_smiles(self, smiles_list):
        if not smiles_list:
            return {}, []

        logger.info("Featurizing %s molecules with MiniMol/Graphium...", len(smiles_list))
        with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
            input_features, failed_indices = self.model.datamodule._featurize_molecules(smiles_list)

        failed_set = set(failed_indices)
        usable = [
            (idx, smiles, feature)
            for idx, (smiles, feature) in enumerate(zip(smiles_list, input_features))
            if idx not in failed_set and not isinstance(feature, str)
        ]
        failed_smiles = [smiles_list[idx] for idx in sorted(failed_set)]

        embeddings = {}
        logger.info("Running MiniMol GNN inference for %s molecules...", len(usable))
        for start in tqdm(range(0, len(usable), self.batch_size), desc="MiniMol batches"):
            batch_records = usable[start : start + self.batch_size]
            batch_features = cast_feature_tensors_to_fp32([record[2] for record in batch_records])
            batch = self.batch_cls.from_data_list(batch_features)
            batch_index_cpu = batch.batch.detach().clone()
            gpu_batch = batch.to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            node_features = self.model.predictor.get_fingerprints_for_batch(
                {"features": gpu_batch, "batch_indices": gpu_batch.batch}
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            fingerprint_graph = self.global_max_pool(node_features, batch_index_cpu)
            fingerprint_graph = fingerprint_graph.detach().cpu().numpy().astype(np.float32, copy=False)
            for row, (_, smiles, _) in zip(fingerprint_graph, batch_records):
                embeddings[smiles] = row

        return embeddings, failed_smiles


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return embeddings / norms


def compute_cosine_similarity_results(
    query_embeddings,
    train_embeddings,
    query_labels_map,
    train_labels_map,
    split_name,
    device,
    chunk_size,
):
    train_smiles = [smiles for smiles in train_embeddings if smiles in train_labels_map]
    query_smiles = [smiles for smiles in query_embeddings if smiles in query_labels_map]

    if not train_smiles:
        raise ValueError("No train MiniMol embeddings overlap with the train label map.")

    train_matrix = normalize_embeddings(
        np.stack([train_embeddings[smiles] for smiles in train_smiles]).astype(np.float32, copy=False)
    )
    query_matrix = normalize_embeddings(
        np.stack([query_embeddings[smiles] for smiles in query_smiles]).astype(np.float32, copy=False)
    )
    train_labels = np.asarray([int(train_labels_map[smiles]) for smiles in train_smiles], dtype=np.int8)
    label_indices = {
        0: np.flatnonzero(train_labels == 0),
        1: np.flatnonzero(train_labels == 1),
    }
    train_index_by_smiles = {smiles: idx for idx, smiles in enumerate(train_smiles)}
    exclude_self = split_name == "train"

    use_torch = device.type == "cuda"
    if use_torch:
        train_tensor = torch.from_numpy(train_matrix).to(device)
    else:
        train_tensor = None

    results = {}
    for start in tqdm(range(0, len(query_smiles), chunk_size), desc=f"{split_name}->train cosine"):
        end = min(start + chunk_size, len(query_smiles))
        if use_torch:
            query_tensor = torch.from_numpy(query_matrix[start:end]).to(device)
            scores_chunk = (query_tensor @ train_tensor.T).detach().cpu().numpy()
        else:
            scores_chunk = query_matrix[start:end] @ train_matrix.T

        for local_idx, scores in enumerate(scores_chunk):
            query_smiles_value = query_smiles[start + local_idx]
            if exclude_self:
                self_idx = train_index_by_smiles.get(query_smiles_value)
                if self_idx is not None:
                    scores = scores.copy()
                    scores[self_idx] = -np.inf

            query_result = {"query_label": int(query_labels_map[query_smiles_value])}
            for label_value, key in ((0, "label_0"), (1, "label_1")):
                idx = label_indices[label_value]
                idx = idx[np.isfinite(scores[idx])]
                sorted_idx = idx[np.argsort(scores[idx])[::-1]]
                query_result[key] = [(float(scores[i]), train_smiles[i]) for i in sorted_idx]
            results[query_smiles_value] = query_result

    return results


def get_tdc_task(task_name, group_name):
    if group_name == "Tox":
        return Tox(name=task_name)
    if group_name == "ADME":
        return ADME(name=task_name)
    if group_name == "HTS":
        return HTS(name=task_name)
    raise ValueError(f"Unknown group {group_name} for task {task_name}")


def process_task(
    task_name,
    group_name,
    requested_splits,
    embedder,
    similarity_device,
    similarity_chunk_size,
    export_clean_data=True,
    export_clean_only=False,
    force_recompute_embeddings=False,
    force_recompute_similarity=False,
):
    logger.info("========== Starting task: %s ==========", task_name)

    base_dir = Path(__file__).parent
    project_root = base_dir.parent.parent
    label_map_dir = base_dir / "Label_maps" / "by_task" / task_name
    conflict_dir = base_dir / "Label_conflicts" / "by_task" / task_name
    clean_export_root = project_root / "DataPrepare" / "TDC_no_conflict_labels_salt_removed"

    label_map_dir.mkdir(parents=True, exist_ok=True)
    conflict_dir.mkdir(parents=True, exist_ok=True)
    if export_clean_data or export_clean_only:
        for split_name in VALID_SPLITS:
            (clean_export_root / split_name).mkdir(parents=True, exist_ok=True)

    data = get_tdc_task(task_name, group_name)
    requested_splits = tuple(dict.fromkeys(requested_splits))
    required_splits = set(requested_splits)
    if any(split_name in requested_splits for split_name in ("valid", "test")):
        required_splits.add("train")

    split = data.get_split(method="scaffold")
    split_dfs = {}
    split_smiles_unique = {}

    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue
        split_df = split[split_name].copy()
        logger.info("Removing salts from %s set SMILES...", split_name)
        split_df["Drug"] = [remove_salts(smiles) for smiles in split_df["Drug"]]
        split_dfs[split_name] = split_df
        split_smiles_unique[split_name] = split_df["Drug"].unique().tolist()
        logger.info("%s - %s SMILES: %s", task_name, split_name.capitalize(), len(split_smiles_unique[split_name]))

    safe_labels_by_split = {}
    for split_name in required_splits:
        analysis = analyze_split_labels(split[split_name], split_dfs[split_name])
        records = analysis["label_map_records"]
        safe_labels_by_split[split_name] = {record["canonical_smiles"]: record["Y"] for record in records}

        if export_clean_data or export_clean_only:
            export_clean_split_records(clean_export_root, task_name, split_name, records)

        with open(label_map_dir / f"{split_name}_labels.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(conflict_dir / f"{split_name}_excluded_conflicts.jsonl", "w", encoding="utf-8") as f:
            for record in analysis["canonical_conflicts"]:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if analysis["canonical_conflicts"]:
            logger.warning(
                "%s %s: excluded %s ambiguous canonical SMILES from label maps and similarity computation.",
                task_name,
                split_name,
                len(analysis["canonical_conflicts"]),
            )

    if export_clean_only:
        logger.info("%s - Exported cleaned no-conflict JSONL files only; skipping MiniMol.", task_name)
        logger.info("========== Finished task: %s ==========\n", task_name)
        return

    minimol_dir = base_dir / "MiniMol" / "by_task" / task_name
    minimol_sim_dir = base_dir / "MiniMol_similarity" / "by_task" / task_name
    minimol_dir.mkdir(parents=True, exist_ok=True)
    minimol_sim_dir.mkdir(parents=True, exist_ok=True)

    embeddings_by_split = {}
    for split_name in VALID_SPLITS:
        if split_name not in required_splits:
            continue
        embed_path = minimol_dir / f"{split_name}.pkl"
        cached = None if force_recompute_embeddings else load_pickle_if_exists(embed_path)
        if cached is not None:
            logger.info("%s - Loading cached MiniMol embeddings for %s split.", task_name, split_name)
            embeddings_by_split[split_name] = cached
            continue

        embeddings, failed_smiles = embedder.embed_smiles(split_smiles_unique[split_name])
        if failed_smiles:
            logger.warning("%s %s: MiniMol failed on %s SMILES.", task_name, split_name, len(failed_smiles))
        embeddings_by_split[split_name] = embeddings
        with open(embed_path, "wb") as f:
            pickle.dump(embeddings, f)

    logger.info("%s - MiniMol embeddings are ready.", task_name)

    train_embeddings = embeddings_by_split["train"]
    train_labels_map = safe_labels_by_split["train"]
    for split_name in requested_splits:
        sim_path = minimol_sim_dir / f"{split_name}_similarity.pkl"
        if sim_path.exists() and not force_recompute_similarity:
            logger.info("%s - MiniMol similarity for %s already exists; skipping.", task_name, split_name)
            continue

        logger.info("Computing %s->Train MiniMol cosine similarity for %s...", split_name.capitalize(), task_name)
        similarity = compute_cosine_similarity_results(
            query_embeddings=embeddings_by_split[split_name],
            train_embeddings=train_embeddings,
            query_labels_map=safe_labels_by_split[split_name],
            train_labels_map=train_labels_map,
            split_name=split_name,
            device=similarity_device,
            chunk_size=similarity_chunk_size,
        )
        with open(sim_path, "wb") as f:
            pickle.dump(similarity, f)

    logger.info("========== Finished task: %s ==========\n", task_name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute MiniMol embeddings and train-based cosine similarities for selected TDC splits."
    )
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=["train", "valid"])
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASK_GROUPS), default=None)
    parser.add_argument("--device", default="auto", help="Embedding device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--similarity-device",
        default="auto",
        help="Similarity matmul device: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="MiniMol GNN inference batch size.")
    parser.add_argument("--similarity-chunk-size", type=int, default=1024)
    parser.add_argument("--featurization-jobs", type=int, default=8)
    parser.add_argument(
        "--featurization-backend",
        default="threading",
        choices=["threading", "loky"],
        help="Use threading with the FP32 patch; loky child processes do not inherit the patch.",
    )
    parser.add_argument("--featurization-batch-size", type=int, default=1024)
    parser.add_argument("--minimol-source", type=Path, default=DEFAULT_MINIMOL_SOURCE)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--force-recompute-similarity", action="store_true")
    parser.add_argument("--export-clean-only", action="store_true")
    parser.add_argument("--no-export-clean-data", action="store_false", dest="export_clean_data")
    parser.set_defaults(export_clean_data=True)
    return parser.parse_args()


def main():
    args = parse_args()
    task_names = args.tasks or [task_name for task_name, _ in TASKS]

    embed_device = resolve_device(args.device)
    similarity_device = resolve_device(args.similarity_device)
    logger.info("Using embedding device=%s, similarity device=%s", embed_device, similarity_device)

    embedder = None
    if not args.export_clean_only:
        embedder = MiniMolEmbedder(
            batch_size=args.batch_size,
            device=embed_device,
            featurization_jobs=args.featurization_jobs,
            featurization_backend=args.featurization_backend,
            featurization_batch_size=args.featurization_batch_size,
            minimol_source=args.minimol_source,
        )

    for task_name in task_names:
        process_task(
            task_name=task_name,
            group_name=TASK_GROUPS[task_name],
            requested_splits=args.splits,
            embedder=embedder,
            similarity_device=similarity_device,
            similarity_chunk_size=args.similarity_chunk_size,
            export_clean_data=args.export_clean_data,
            export_clean_only=args.export_clean_only,
            force_recompute_embeddings=args.force_recompute_embeddings,
            force_recompute_similarity=args.force_recompute_similarity,
        )


if __name__ == "__main__":
    main()
