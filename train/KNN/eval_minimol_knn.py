import argparse
import csv
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TASKS = [
    "Carcinogens_Lagunin",
    "BBB_Martins",
    "DILI",
    "Pgp_Broccatelli",
    "PAMPA_NCATS",
    "HIA_Hou",
    "Bioavailability_Ma",
    "hERG",
    "AMES",
    "Skin_Reaction",
    "ClinTox",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "SARSCoV2_3CLPro_Diamond",
    "SARSCoV2_Vitro_Touret",
]

SPLITS = ("train", "valid", "test")


def load_minimol_similarity(task_name, split_name):
    base_dir = Path(__file__).parent.parent.parent / "DataPrepare" / "TDC_mol_fingerprints"
    sim_path = base_dir / "MiniMol_similarity" / "by_task" / task_name / f"{split_name}_similarity.pkl"
    if not sim_path.exists():
        logger.warning("Missing MiniMol similarity file: %s", sim_path)
        return None
    with open(sim_path, "rb") as f:
        return pickle.load(f)


def iter_top_k_neighbors(sim_data, k):
    label_0 = sim_data.get("label_0", [])
    label_1 = sim_data.get("label_1", [])
    i0 = 0
    i1 = 0
    emitted = 0

    while emitted < k and (i0 < len(label_0) or i1 < len(label_1)):
        score_0 = label_0[i0][0] if i0 < len(label_0) else -np.inf
        score_1 = label_1[i1][0] if i1 < len(label_1) else -np.inf
        if score_1 >= score_0:
            yield score_1, 1
            i1 += 1
        else:
            yield score_0, 0
            i0 += 1
        emitted += 1


def predict_minimol_knn(sims, k):
    y_true = []
    y_pred = []
    predictions = {}

    for query_smiles, sim_data in tqdm(sims.items(), desc=f"MiniMol KNN K={k}", leave=False):
        true_label = int(sim_data["query_label"])
        votes = [label for _, label in iter_top_k_neighbors(sim_data, k)]
        pred_label = 1 if votes.count(1) >= votes.count(0) else 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        predictions[query_smiles] = pred_label

    return y_true, y_pred, predictions


def compute_metrics(y_true, y_pred):
    metrics = {
        "num_samples": len(y_true),
        "num_positive_true": int(sum(y_true)),
        "num_positive_pred": int(sum(y_pred)),
        "accuracy": None,
        "macro_f1": None,
        "mean_pred": None,
        "per_class": None,
    }
    if not y_true:
        return metrics

    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    metrics["accuracy"] = float(correct / total)
    metrics["mean_pred"] = float(np.mean(y_pred))
    if len(set(y_true)) > 1:
        per_class = {}
        f1_values = []
        for label in (0, 1):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            per_class[str(label)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for t in y_true if t == label),
            }
            f1_values.append(f1)
        metrics["macro_f1"] = float(sum(f1_values) / len(f1_values))
        metrics["per_class"] = per_class
    return metrics


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "k",
        "split",
        "num_samples",
        "num_positive_true",
        "num_positive_pred",
        "accuracy",
        "macro_f1",
        "mean_pred",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_average_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "k",
        "split",
        "num_tasks",
        "avg_accuracy",
        "avg_macro_f1",
        "avg_mean_pred",
        "total_samples",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_average_rows(summary_rows, k_values, splits):
    average_rows = []
    for k in k_values:
        for split_name in splits:
            rows = [row for row in summary_rows if row["k"] == k and row["split"] == split_name]
            if not rows:
                continue
            macro_f1_values = [row["macro_f1"] for row in rows if row["macro_f1"] is not None]
            accuracy_values = [row["accuracy"] for row in rows if row["accuracy"] is not None]
            mean_pred_values = [row["mean_pred"] for row in rows if row["mean_pred"] is not None]
            average_rows.append(
                {
                    "k": k,
                    "split": split_name,
                    "num_tasks": len(rows),
                    "avg_accuracy": float(np.mean(accuracy_values)) if accuracy_values else None,
                    "avg_macro_f1": float(np.mean(macro_f1_values)) if macro_f1_values else None,
                    "avg_mean_pred": float(np.mean(mean_pred_values)) if mean_pred_values else None,
                    "total_samples": sum(row["num_samples"] for row in rows),
                }
            )
    return average_rows


def format_metric(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN using MiniMol cosine similarity files.")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=["train", "valid", "test"])
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent
        / "DataPrepare"
        / "TDC_mol_fingerprints"
        / "MiniMol_KNN",
    )
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    summary_rows = []
    for task_name in args.tasks:
        logger.info("\n%s", "=" * 80)
        logger.info("MiniMol KNN Task: %s", task_name)
        logger.info("%s", "=" * 80)

        valid_sims = load_minimol_similarity(task_name, "valid")
        if valid_sims is None:
            continue

        sims_by_split = {"valid": valid_sims}
        for split_name in args.splits:
            if split_name == "valid":
                continue
            sims_by_split[split_name] = load_minimol_similarity(task_name, split_name)

        task_metrics = {"k_values": {}}
        predictions_by_k = {}
        for k in args.k_values:
            task_metrics["k_values"][str(k)] = {"splits": {}}
            predictions_by_k[k] = {}

            for split_name in args.splits:
                sims = sims_by_split.get(split_name)
                if sims is None:
                    task_metrics["k_values"][str(k)]["splits"][split_name] = None
                    continue

                y_true, y_pred, predictions = predict_minimol_knn(sims, k)
                metrics = compute_metrics(y_true, y_pred)
                task_metrics["k_values"][str(k)]["splits"][split_name] = metrics
                predictions_by_k[k][split_name] = predictions

                logger.info(
                    "%s k=%s %s: n=%s acc=%s macro_f1=%s positives true/pred=%s/%s",
                    task_name,
                    k,
                    split_name,
                    metrics["num_samples"],
                    format_metric(metrics["accuracy"]),
                    format_metric(metrics["macro_f1"]),
                    metrics["num_positive_true"],
                    metrics["num_positive_pred"],
                )

                summary_rows.append(
                    {
                        "task": task_name,
                        "k": k,
                        "split": split_name,
                        "num_samples": metrics["num_samples"],
                        "num_positive_true": metrics["num_positive_true"],
                        "num_positive_pred": metrics["num_positive_pred"],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "mean_pred": metrics["mean_pred"],
                    }
                )

        metrics_path = args.save_dir / "metrics" / "by_task" / task_name / "metrics_by_k.json"
        save_json(metrics_path, task_metrics)
        if args.save_predictions:
            for k, predictions_by_split in predictions_by_k.items():
                for split_name, predictions in predictions_by_split.items():
                    pred_path = (
                        args.save_dir
                        / f"k{k}"
                        / split_name
                        / "by_task"
                        / task_name
                        / f"{split_name}_knn_labels.json"
                    )
                    save_json(pred_path, predictions)

    summary_path = args.save_dir / "metrics" / "summary.csv"
    save_summary_csv(summary_path, summary_rows)
    logger.info("Saved MiniMol KNN summary to %s", summary_path)

    average_rows = compute_average_rows(summary_rows, args.k_values, args.splits)
    average_path = args.save_dir / "metrics" / "average_by_k.csv"
    save_average_csv(average_path, average_rows)
    logger.info("Saved MiniMol KNN average metrics to %s", average_path)

    logger.info("Average performance:")
    for row in average_rows:
        logger.info(
            "k=%-2s %-5s avg_acc=%s avg_macro_f1=%s tasks=%s",
            row["k"],
            row["split"],
            format_metric(row["avg_accuracy"]),
            format_metric(row["avg_macro_f1"]),
            row["num_tasks"],
        )


if __name__ == "__main__":
    main()
