# KNN Baseline for TDC Molecule Tasks

This directory contains the KNN baseline that consumes the precomputed artifacts under `DataPrepare/TDC_mol_fingerprints/`. Its job is not to build fingerprints itself, but to:

1. read precomputed train/valid similarity files,
2. run a simple weighted KNN classifier,
3. evaluate validation performance,
4. export predicted labels for downstream prompt construction.

## Relationship to `DataPrepare/TDC_mol_fingerprints`

The dependency chain is:

1. `DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py`
   creates canonicalized label maps, fingerprint pickles, and train/valid similarity pickles.
2. `train/KNN/eval_knn.py`
   reads those similarity pickles and reports baseline KNN metrics on `valid`.
3. `train/KNN/extract_best_knn_labels.py`
   selects the best `k` on `valid`, then exports `train` and `valid` pseudo labels.
4. `DataPrepare/TDC_prepended/generate_knn_prompts.py`
   reads the pseudo labels and similarity files to build KNN-augmented prompts.

The KNN score is a weighted sum of two similarity sources:

```text
weighted_score = 0.8 * Morgan_similarity + 0.2 * Feature_Morgan_similarity
```

For each query molecule, the top-`k` references are selected after weighting, and the final prediction is produced by majority vote. Ties go to label `1`.

## Files in This Directory

### `eval_knn.py`

Validation-only evaluator.

What it does:

- loads `valid_similarity.pkl` from both `Morgan_similarity` and `Feature_Morgan_similarity`,
- combines the two scores with weights `0.8` and `0.2`,
- runs majority-vote KNN on the validation queries,
- prints `classification_report` and macro-F1.

Inputs:

- `DataPrepare/TDC_mol_fingerprints/Morgan_similarity/by_task/<task>/valid_similarity.pkl`
- `DataPrepare/TDC_mol_fingerprints/Feature_Morgan_similarity/by_task/<task>/valid_similarity.pkl`

Example:

```bash
python train/KNN/eval_knn.py --tasks BBB_Martins DILI Pgp_Broccatelli -k 3
```

Arguments:

- `--tasks`: one or more task names.
- `-k`: number of neighbors, default `3`.

### `extract_best_knn_labels.py`

Pseudo-label extraction pipeline.

What it does:

- tries candidate `k` values in `{3, 6, 9, 12}` on the validation split,
- chooses the best `k` by macro-F1,
- re-runs KNN on `train` and `valid` using that best `k`,
- saves predicted labels as JSON maps from canonical SMILES to predicted class.

Inputs:

- `DataPrepare/TDC_mol_fingerprints/Morgan_similarity/by_task/<task>/{train,valid}_similarity.pkl`
- `DataPrepare/TDC_mol_fingerprints/Feature_Morgan_similarity/by_task/<task>/{train,valid}_similarity.pkl`

Outputs:

- `DataPrepare/TDC_mol_fingerprints/KNN_pesudo_labels/by_task/<task>/train_knn_labels.json`
- `DataPrepare/TDC_mol_fingerprints/KNN_pesudo_labels/by_task/<task>/valid_knn_labels.json`

Note:

- The output directory name is spelled `KNN_pesudo_labels` in code and on disk.
- The `tasks` list is currently hard-coded inside the script, so you usually edit that list before running it.

Example:

```bash
python train/KNN/extract_best_knn_labels.py
```

## Recommended Workflow

If you are starting from raw TDC tasks:

```bash
python DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py
python train/KNN/eval_knn.py --tasks Carcinogens_Lagunin -k 3
python train/KNN/extract_best_knn_labels.py
python DataPrepare/TDC_prepended/generate_knn_prompts.py --tasks Carcinogens_Lagunin --splits train valid --top-k 3
```

If the similarity files already exist and you only want KNN outputs:

```bash
python train/KNN/eval_knn.py --tasks Carcinogens_Lagunin -k 3
python train/KNN/extract_best_knn_labels.py
```

## Data Format Expected by These Scripts

Each similarity pickle is a dictionary:

```python
{
    query_smiles: {
        "query_label": 0 or 1,
        "label_0": [(score, ref_smiles), ...],  # sorted descending
        "label_1": [(score, ref_smiles), ...],  # sorted descending
    }
}
```

`eval_knn.py` and `extract_best_knn_labels.py` rebuild a flat reference-score table from those two label-specific lists, then rank all references by the weighted score.

## Practical Notes

- These scripts assume the SMILES keys already match the canonicalized/desalted form generated in `DataPrepare/TDC_mol_fingerprints`.
- `train_similarity.pkl` is computed with self-matches removed, which matters when generating pseudo labels for the training split.
- If a task has only one class in `valid`, macro-F1 cannot be computed and the script logs that situation instead.
