import os
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

def load_similarities(task_name):
    base_dir = Path(__file__).parent.parent.parent / "DataPrepare" / "TDC_mol_fingerprints"
    
    morgan_sim_path = base_dir / "Morgan_similarity" / "by_task" / task_name / "valid_similarity.pkl"
    feat_morgan_sim_path = base_dir / "Feature_Morgan_similarity" / "by_task" / task_name / "valid_similarity.pkl"
    
    if not morgan_sim_path.exists() or not feat_morgan_sim_path.exists():
        logger.error(f"Cannot find precomputed similarity files for task {task_name}.")
        return None, None
        
    logger.info(f"Loading Morgan similarity for {task_name}...")
    with open(morgan_sim_path, "rb") as f:
        morgan_sims = pickle.load(f)
        
    logger.info(f"Loading Feature Morgan similarity for {task_name}...")
    with open(feat_morgan_sim_path, "rb") as f:
        feat_morgan_sims = pickle.load(f)
        
    return morgan_sims, feat_morgan_sims

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
        
        # We need to map reference smiles to their similarity scores for both fp types
        # so we can compute the weighted sum later.
        
        # Helper inner function to parse scores into a flat dict mapping ref_smiles -> {"score": float, "label": int}
        def parse_to_dict(sim_data):
            ref_dict = {}
            for label_name, label_val in [("label_0", 0), ("label_1", 1)]:
                # Each list is composed of (score, ref_sm) tuples
                for score, ref_sm in sim_data[label_name]:
                    ref_dict[ref_sm] = {"score": score, "label": label_val}
            return ref_dict
            
        m_refs = parse_to_dict(m_data)
        f_refs = parse_to_dict(f_data)
        
        # Calculate weighted similarities for all reference molecules
        weighted_scores = []
        all_ref_smiles = set(list(m_refs.keys()) + list(f_refs.keys()))
        for ref_smiles in all_ref_smiles:
            m_score = m_refs.get(ref_smiles, {}).get("score", 0.0)
            f_score = f_refs.get(ref_smiles, {}).get("score", 0.0)
            
            # Label is identical in both lists, grab it from whichever has it
            ref_label = m_refs.get(ref_smiles, {}).get("label")
            if ref_label is None:
                ref_label = f_refs.get(ref_smiles, {}).get("label")
            
            final_score = (w_morgan * m_score) + (w_feat * f_score)
            weighted_scores.append((final_score, ref_label))
            
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

def main():
    parser = argparse.ArgumentParser(description='Run baseline KNN classifier using derived Molecule fingerpints similarities')
    parser.add_argument('--tasks', nargs='+', default=['BBB_Martins', 'DILI', 'Pgp_Broccatelli'], help='Tasks to evaluate')
    parser.add_argument('-k', type=int, default=3, help='Number of nearest neighbors')
    
    args = parser.parse_args()
    
    for task_name in args.tasks:
        logger.info("\n" + "=" * 80)
        logger.info(f"{task_name} KNN EVALUATION (K={args.k})")
        logger.info("=" * 80)
        
        morgan_sims, feat_morgan_sims = load_similarities(task_name)
        if morgan_sims is None or feat_morgan_sims is None:
            continue
            
        y_true, y_pred = predict_knn(morgan_sims, feat_morgan_sims, k=args.k)
        
        if len(set(y_true)) > 1:
            score = f1_score(y_true, y_pred, average='macro', pos_label=1)
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y_true, y_pred, digits=4, labels=[0, 1]))
            logger.info(f"F1 Score: {score:.4f}")
        else:
            logger.info(f"{task_name} F1: Cannot calculate (one class only in truth)")
            logger.info(f"Mean Pred: {np.mean(y_pred):.4f}")
            
        logger.info("=" * 80 + "\n")

if __name__ == "__main__":
    main()
