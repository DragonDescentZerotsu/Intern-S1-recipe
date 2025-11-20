# Simple Baseline

This file documents the **simple baseline** used in our experiments.  
For each TDC binary classification task, we train a **majority class classifier** that always predicts the most frequent label in the training set (i.e., all-0 or all-1 predictions on the test set).

Because this classifier assigns the **same score to all samples**, it has **no ranking ability**. Under the AUROC metric, such a non-informative classifier corresponds to **chance-level performance**, i.e. an AUROC of **0.5** for all tasks.

To run the simplre baseline, execute the following command:
```bash
python simple_baseline.py
```

The resulting **group-average AUROC** scores are:

- ADME: 0.5000  
- Tox: 0.5000  
- HTS: 0.5000  
- Develop: 0.5000  
- PPI: 0.5000  
- TCREpitopeBinding: 0.5000  
- TrialOutcome: 0.5000  
- PeptideMHC: 0.5000  

This simple baseline serves as a **chance-level reference point**; any practical model should outperform these AUROC ≈ 0.5 scores.
