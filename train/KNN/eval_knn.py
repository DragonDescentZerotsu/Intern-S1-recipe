import argparse
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def load_similarities(task_name, split_name):
    base_dir = Path(__file__).parent.parent.parent / "DataPrepare" / "TDC_mol_fingerprints"
    
    morgan_sim_path = base_dir / "Morgan_similarity" / "by_task" / task_name / f"{split_name}_similarity.pkl"
    feat_morgan_sim_path = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name / f"{split_name}_similarity.pkl"
    
    if not morgan_sim_path.exists() or not feat_morgan_sim_path.exists():
        logger.error(f"Cannot find precomputed {split_name} similarity files for task {task_name}.")
        return None, None
        
    logger.info(f"Loading Morgan {split_name}->train similarity for {task_name}...")
    with open(morgan_sim_path, "rb") as f:
        morgan_sims = pickle.load(f)
        
    logger.info(f"Loading Feature Morgan {split_name}->train similarity for {task_name}...")
    with open(feat_morgan_sim_path, "rb") as f:
        feat_morgan_sims = pickle.load(f)
        
    return morgan_sims, feat_morgan_sims

def parse_to_dict(sim_data):
    ref_dict = {}
    for label_name, label_val in [("label_0", 0), ("label_1", 1)]:
        # Each list is composed of (score, ref_sm) tuples.
        for score, ref_sm in sim_data.get(label_name, []):
            ref_dict[ref_sm] = {"score": score, "label": label_val}
    return ref_dict

def get_weighted_scores(m_data, f_data, w_morgan=0.8, w_feat=0.2):
    # Map reference smiles to their similarity scores for both fingerprint types.
    m_refs = parse_to_dict(m_data)
    f_refs = parse_to_dict(f_data)

    weighted_scores = []
    all_ref_smiles = set(list(m_refs.keys()) + list(f_refs.keys()))
    for ref_smiles in all_ref_smiles:
        m_score = m_refs.get(ref_smiles, {}).get("score", 0.0)
        f_score = f_refs.get(ref_smiles, {}).get("score", 0.0)

        # Label is identical in both lists, grab it from whichever has it.
        ref_label = m_refs.get(ref_smiles, {}).get("label")
        if ref_label is None:
            ref_label = f_refs.get(ref_smiles, {}).get("label")

        final_score = (w_morgan * m_score) + (w_feat * f_score)
        weighted_scores.append((final_score, ref_label))

    return weighted_scores

def predict_knn(morgan_sims, feat_morgan_sims, k=3, w_morgan=0.8, w_feat=0.2):
    y_true = []
    y_pred = []
    
    # Iterate through all validation queries
    for query_smiles in tqdm(morgan_sims.keys(), desc="Evaluating KNN"):
        m_data = morgan_sims[query_smiles]
        f_data = feat_morgan_sims[query_smiles]
        
        # Ground truth is explicitly stored at root dimension
        true_label = m_data['query_label']
        y_true.append(true_label)

        weighted_scores = get_weighted_scores(m_data, f_data, w_morgan=w_morgan, w_feat=w_feat)
            
        # Sort descending by weighted score
        weighted_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Take Top K
        top_k = weighted_scores[:k]
        
        # Majority Voting among top K
        votes = [x[1] for x in top_k]
        if votes.count(1) >= votes.count(0):
            predicted_label = 1
        else:
            predicted_label = 0
            
        y_pred.append(predicted_label)
        
    return y_true, y_pred

def predict_balanced_similarity_knn(
    morgan_sims,
    feat_morgan_sims,
    k=4,
    w_morgan=0.8,
    w_feat=0.2,
    tie_label=1,
):
    y_true = []
    y_pred = []
    scores = []

    for query_smiles in tqdm(morgan_sims.keys(), desc="Evaluating balanced similarity KNN"):
        m_data = morgan_sims[query_smiles]
        f_data = feat_morgan_sims[query_smiles]

        true_label = int(m_data["query_label"])
        y_true.append(true_label)

        weighted_scores = get_weighted_scores(m_data, f_data, w_morgan=w_morgan, w_feat=w_feat)
        pos_scores = sorted(
            (score for score, label in weighted_scores if label == 1),
            reverse=True,
        )[:k]
        neg_scores = sorted(
            (score for score, label in weighted_scores if label == 0),
            reverse=True,
        )[:k]

        query_score = float(sum(pos_scores) - sum(neg_scores))
        if query_score > 0:
            predicted_label = 1
        elif query_score < 0:
            predicted_label = 0
        else:
            predicted_label = tie_label

        y_pred.append(predicted_label)
        scores.append(query_score)

    return y_true, y_pred, scores

def evaluate_predictions(task_name, split_name, y_true, y_pred):
    result = {
        "task": task_name,
        "split": split_name,
        "num_samples": len(y_true),
        "num_positive_true": int(sum(y_true)),
        "num_positive_pred": int(sum(y_pred)),
        "accuracy": None,
        "macro_f1": None,
    }
    if not y_true:
        logger.info(f"{task_name} {split_name}: no samples")
        return result

    result["accuracy"] = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))
    if len(set(y_true)) > 1:
        result["macro_f1"] = float(f1_score(y_true, y_pred, average='macro', pos_label=1))
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_true, y_pred, digits=4, labels=[0, 1]))
        logger.info(
            f"{task_name} {split_name}: "
            f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
            f"positives true/pred={result['num_positive_true']}/{result['num_positive_pred']}"
        )
    else:
        logger.info(f"{task_name} {split_name} F1: Cannot calculate (one class only in truth)")
        logger.info(f"{task_name} {split_name} Mean Pred: {np.mean(y_pred):.4f}")

    return result

def format_metric(value):
    if value is None:
        return "NA"
    return f"{value:.4f}"

def main():
    parser = argparse.ArgumentParser(description='Run baseline KNN classifier using derived Molecule fingerpints similarities')
    parser.add_argument('--tasks', nargs='+', default=TASKS, help='Tasks to evaluate')
    parser.add_argument('--splits', nargs='+', choices=SPLITS, default=['valid', 'test'], help='Splits to evaluate')
    parser.add_argument('-k', type=int, default=3, help='Number of nearest neighbors')
    parser.add_argument(
        '--mode',
        choices=['majority', 'balanced-similarity'],
        default='majority',
        help='Prediction rule: original top-k majority vote, or top-k per label signed similarity score',
    )
    parser.add_argument('--tie-label', type=int, choices=[0, 1], default=1, help='Prediction when balanced similarity score is exactly 0')
    
    args = parser.parse_args()

    all_results = []
    for task_name in args.tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"{task_name} KNN EVALUATION (K={args.k}, mode={args.mode})")
        logger.info("=" * 80)

        for split_name in args.splits:
            logger.info(f"\n{task_name} split={split_name} (queries use train-set neighbors)")
            morgan_sims, feat_morgan_sims = load_similarities(task_name, split_name)
            if morgan_sims is None or feat_morgan_sims is None:
                continue

            if args.mode == 'majority':
                y_true, y_pred = predict_knn(morgan_sims, feat_morgan_sims, k=args.k)
            else:
                y_true, y_pred, _ = predict_balanced_similarity_knn(
                    morgan_sims,
                    feat_morgan_sims,
                    k=args.k,
                    tie_label=args.tie_label,
                )

            all_results.append(evaluate_predictions(task_name, split_name, y_true, y_pred))

        logger.info("=" * 80 + "\n")

    logger.info("\nAverage performance:")
    for split_name in args.splits:
        split_results = [r for r in all_results if r["split"] == split_name]
        if not split_results:
            continue
        macro_f1_values = [r["macro_f1"] for r in split_results if r["macro_f1"] is not None]
        accuracy_values = [r["accuracy"] for r in split_results if r["accuracy"] is not None]
        logger.info(
            "%s: tasks=%s avg_accuracy=%s avg_macro_f1=%s",
            split_name,
            len(split_results),
            format_metric(float(np.mean(accuracy_values)) if accuracy_values else None),
            format_metric(float(np.mean(macro_f1_values)) if macro_f1_values else None),
        )

if __name__ == "__main__":
    main()
