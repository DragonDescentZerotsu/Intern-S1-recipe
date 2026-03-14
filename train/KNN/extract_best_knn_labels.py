import os
import argparse
import pickle
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_similarities(task_name, split_name):
    base_dir = Path(__file__).parent.parent.parent / "DataPrepare" / "TDC_mol_fingerprints"
    
    morgan_sim_path = base_dir / "Morgan_similarity" / "by_task" / task_name / f"{split_name}_similarity.pkl"
    feat_morgan_sim_path = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name / f"{split_name}_similarity.pkl"
    
    if not morgan_sim_path.exists() or not feat_morgan_sim_path.exists():
        logger.error(f"Cannot find precomputed {split_name} similarity files for task {task_name}.")
        return None, None
        
    logger.info(f"Loading {split_name} Morgan similarity for {task_name}...")
    with open(morgan_sim_path, "rb") as f:
        morgan_sims = pickle.load(f)
        
    logger.info(f"Loading {split_name} Feature Morgan similarity for {task_name}...")
    with open(feat_morgan_sim_path, "rb") as f:
        feat_morgan_sims = pickle.load(f)
        
    return morgan_sims, feat_morgan_sims

def predict_knn(morgan_sims, feat_morgan_sims, k=3, w_morgan=0.8, w_feat=0.2):
    y_true = []
    y_pred = []
    predictions_dict = {}
    
    for query_smiles in tqdm(morgan_sims.keys(), desc=f"Evaluating KNN (K={k})", leave=False):
        m_data = morgan_sims[query_smiles]
        f_data = feat_morgan_sims[query_smiles]
        
        true_label = m_data['query_label']
        y_true.append(true_label)
        
        def parse_to_dict(sim_data):
            ref_dict = {}
            for label_name, label_val in [("label_0", 0), ("label_1", 1)]:
                for score, ref_sm in sim_data[label_name]:
                    ref_dict[ref_sm] = {"score": score, "label": label_val}
            return ref_dict
            
        m_refs = parse_to_dict(m_data)
        f_refs = parse_to_dict(f_data)
        
        weighted_scores = []
        all_ref_smiles = set(list(m_refs.keys()) + list(f_refs.keys()))
        for ref_smiles in all_ref_smiles:
            m_score = m_refs.get(ref_smiles, {}).get("score", 0.0)
            f_score = f_refs.get(ref_smiles, {}).get("score", 0.0)
            
            ref_label = m_refs.get(ref_smiles, {}).get("label")
            if ref_label is None:
                ref_label = f_refs.get(ref_smiles, {}).get("label")
            
            final_score = (w_morgan * m_score) + (w_feat * f_score)
            weighted_scores.append((final_score, ref_label))
            
        weighted_scores.sort(key=lambda x: x[0], reverse=True)
        top_k = weighted_scores[:k]
        
        votes = [x[1] for x in top_k]
        predicted_label = 1 if votes.count(1) >= votes.count(0) else 0
            
        y_pred.append(predicted_label)
        predictions_dict[query_smiles] = predicted_label
        
    return y_true, y_pred, predictions_dict


def save_predictions(save_path, predictions_dict, split_name):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(predictions_dict, f, indent=4)
    logger.info(f"Saved {len(predictions_dict)} {split_name} labels to {save_path}")


def report_split_performance(task_name, split_name, y_true, y_pred):
    metrics = {
        "num_samples": len(y_true),
        "accuracy": None,
        "macro_f1": None,
        "mean_pred": None,
        "classification_report": None,
    }

    if not y_true:
        logger.warning("%s %s: no samples available for evaluation.", task_name, split_name)
        return metrics

    accuracy = accuracy_score(y_true, y_pred)
    metrics["accuracy"] = accuracy
    logger.info("%s %s Accuracy: %.4f", task_name, split_name.capitalize(), accuracy)

    if len(set(y_true)) > 1:
        macro_f1 = f1_score(y_true, y_pred, average='macro', pos_label=1)
        report = classification_report(y_true, y_pred, digits=4, labels=[0, 1])
        metrics["macro_f1"] = macro_f1
        metrics["classification_report"] = report
        logger.info("%s %s Macro-F1: %.4f", task_name, split_name.capitalize(), macro_f1)
        logger.info(
            "%s %s Classification Report:\n%s",
            task_name,
            split_name.capitalize(),
            report,
        )
    else:
        mean_pred = float(np.mean(y_pred))
        metrics["mean_pred"] = mean_pred
        logger.info(
            "%s %s Macro-F1: Cannot calculate (one class only in truth)",
            task_name,
            split_name.capitalize(),
        )
        logger.info("%s %s Mean Pred: %.4f", task_name, split_name.capitalize(), mean_pred)

    return metrics


def save_metrics(save_path, metrics_dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Saved metrics to {save_path}")

def main():
    tasks = [
        # 'Carcinogens_Lagunin',
        'BBB_Martins',
        'DILI',
        'Pgp_Broccatelli'
        # 'PAMPA_NCATS',
        # 'HIA_Hou',
        # 'Bioavailability_Ma',
        # 'hERG',
        # 'AMES',
        # 'Skin_Reaction',
        # 'ClinTox',
        # 'CYP2C9_Substrate_CarbonMangels',
        # 'CYP2D6_Substrate_CarbonMangels',
        # 'CYP3A4_Substrate_CarbonMangels',
        # 'SARSCoV2_3CLPro_Diamond',
        # 'SARSCoV2_Vitro_Touret',
    ]
    k_candidates = [3, 6, 9, 12]
    
    base_dir = Path(__file__).parent.parent.parent / "DataPrepare" / "TDC_mol_fingerprints"
    save_base_dir = base_dir / "KNN_pesudo_labels"
    metrics_base_dir = save_base_dir / "metrics" / "by_task"
    
    for task_name in tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"Processing Task: {task_name}")
        logger.info("=" * 80)
        
        # 1. Evaluate on validation set to find the best K
        morgan_sims_valid, feat_morgan_sims_valid = load_similarities(task_name, "valid")
        if not morgan_sims_valid:
            continue
            
        best_k = -1
        best_f1 = -1.0
        best_valid_y_true = []
        best_valid_y_pred = []
        best_valid_preds = {}
        
        for k in k_candidates:
            y_true, y_pred, preds_dict = predict_knn(morgan_sims_valid, feat_morgan_sims_valid, k=k)
            if len(set(y_true)) > 1:
                score = f1_score(y_true, y_pred, average='macro', pos_label=1)
                logger.info(f"Validation F1 for K={k:2d}: {score:.4f}")
                if score > best_f1:
                    best_f1 = score
                    best_k = k
                    best_valid_preds = preds_dict
            else:
                logger.info(f"Validation F1 for K={k:2d}: Cannot calculate (one class only in truth)")

        if best_k == -1:
            best_k = k_candidates[0]
            best_valid_y_true, best_valid_y_pred, best_valid_preds = predict_knn(
                morgan_sims_valid,
                feat_morgan_sims_valid,
                k=best_k,
            )
            logger.info(
                "\n=> %s valid has one class only; falling back to K=%d for prediction/export.",
                task_name,
                best_k,
            )
        else:
            best_valid_y_true, best_valid_y_pred, best_valid_preds = predict_knn(
                morgan_sims_valid,
                feat_morgan_sims_valid,
                k=best_k,
            )

        logger.info(f"\n=> Best K for {task_name} is {best_k} (F1 = {best_f1:.4f})")
        task_metrics = {
            "best_k": best_k,
            "validation_selection_macro_f1": None if best_f1 < 0 else best_f1,
            "splits": {},
        }
        task_metrics["splits"]["valid"] = report_split_performance(
            task_name,
            "valid",
            best_valid_y_true,
            best_valid_y_pred,
        )
        
        # 2. Get predictions for train/valid/test using the best K
        morgan_sims_train, feat_morgan_sims_train = load_similarities(task_name, "train")
        train_y_true, train_y_pred, train_preds = predict_knn(morgan_sims_train, feat_morgan_sims_train, k=best_k)
        task_metrics["splits"]["train"] = report_split_performance(task_name, "train", train_y_true, train_y_pred)

        morgan_sims_test, feat_morgan_sims_test = load_similarities(task_name, "test")
        test_preds = None
        if morgan_sims_test and feat_morgan_sims_test:
            test_y_true, test_y_pred, test_preds = predict_knn(morgan_sims_test, feat_morgan_sims_test, k=best_k)
            task_metrics["splits"]["test"] = report_split_performance(task_name, "test", test_y_true, test_y_pred)
        else:
            task_metrics["splits"]["test"] = None
            logger.warning(f"Skipping test pseudo labels for {task_name} because test similarity files are missing.")

        # 3. Save to files
        train_save_path = save_base_dir / "train" / "by_task" / task_name / "train_knn_labels.json"
        valid_save_path = save_base_dir / "valid" / "by_task" / task_name / "valid_knn_labels.json"

        save_predictions(valid_save_path, best_valid_preds, "valid")
        save_predictions(train_save_path, train_preds, "train")

        if test_preds is not None:
            test_save_path = save_base_dir / "test" / "by_task" / task_name / "test_knn_labels.json"
            save_predictions(test_save_path, test_preds, "test")

        metrics_save_path = metrics_base_dir / task_name / "metrics.json"
        save_metrics(metrics_save_path, task_metrics)

if __name__ == "__main__":
    main()
