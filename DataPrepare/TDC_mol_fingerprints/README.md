# TDC Molecule Fingerprints and Similarity Artifacts

This directory is the data-preparation side of the KNN pipeline. It turns TDC molecule classification tasks into reusable artifacts that later scripts consume.

The main outputs here are:

- canonicalized label maps,
- conflict reports for ambiguous labels,
- Morgan fingerprints,
- Feature Morgan fingerprints,
- train-to-train and valid-to-train similarity pickles,
- KNN pseudo labels generated later by `train/KNN/extract_best_knn_labels.py`.

## How This Directory Connects to the Rest of the Project

The full pipeline is:

1. `compute_fingerprints_and_similarities.py`
   loads TDC data, removes salts, canonicalizes SMILES, filters ambiguous labels, and computes fingerprints and similarity files.
2. `train/KNN/eval_knn.py`
   evaluates a weighted KNN baseline using the similarity files created here.
3. `train/KNN/extract_best_knn_labels.py`
   writes pseudo labels back into this directory under `KNN_pesudo_labels/`.
4. `DataPrepare/TDC_prepended/generate_knn_prompts.py`
   consumes the similarity files and pseudo labels to build prompt datasets.

So this folder is the shared storage layer between raw TDC data and downstream KNN / prompt-generation scripts.

## Directory Layout

### `compute_fingerprints_and_similarities.py`

Core preprocessing script.

What it does for each task:

1. loads the task from TDC (`ADME`, `Tox`, or `HTS`),
2. gets the scaffold split and keeps `train` and `valid`,
3. removes salts with `LargestFragmentChooser`,
4. computes canonical label maps and excludes ambiguous canonical SMILES,
5. computes two fingerprint types:
   - standard Morgan fingerprint,
   - Feature Morgan fingerprint,
6. computes similarity files:
   - `valid -> train`,
   - `train -> train` with self-match removed,
7. saves all artifacts to disk.

Important implementation details:

- fingerprint settings are `radius=2`, `nBits=2048`,
- Tanimoto similarity is used,
- similarity results are stored separately for reference label `0` and label `1`,
- ambiguous labels introduced by canonicalization are written to `Label_conflicts/` and excluded from later similarity computation.

Before running it, you usually edit the hard-coded `tasks` list near the bottom of the file.

Example:

```bash
python DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py
```

### `compute_jsonl_fingerprints_and_similarities.py`

Local JSONL entry point that reuses the same RDKit Morgan and similarity helpers as the TDC script, but reads existing `train.jsonl`, `valid.jsonl`, and `test.jsonl` files with `drug` and `Y` fields.

It writes the same artifact layout under `DataPrepare/TDC_mol_fingerprints/*/by_task/<task_name>/`, so downstream KNN code can consume the outputs by task name.

Example for the high-confidence B3DB-rescued BBB_Martins split:

```bash
/data1/tianang/anaconda3/condabin/conda run -n vllm python DataPrepare/TDC_mol_fingerprints/compute_jsonl_fingerprints_and_similarities.py --num-cpus 8
```

### `verify_sims.py`

Small sanity-check utility for one example task.

What it checks:

- similarity pickle files can be loaded,
- entries are sorted in descending similarity order,
- `train_similarity.pkl` excludes self-matches.

This is useful after regenerating similarity files and before trusting them downstream.

Example:

```bash
python DataPrepare/TDC_mol_fingerprints/verify_sims.py
```

### `visualize.py`

Quick inspection tool for pickle outputs.

What it does:

- opens a `.pkl` file,
- prints the overall structure,
- shows a few sample keys and example values,
- handles both fingerprint dictionaries and similarity dictionaries.

Example:

```bash
python DataPrepare/TDC_mol_fingerprints/visualize.py DataPrepare/TDC_mol_fingerprints/Feature_Morgan_similarity/by_task/BBB_Martins/valid_similarity.pkl 2
```

## Artifact Directories

### `Morgan/by_task/<task>/`

Stores pickled dictionaries of standard Morgan fingerprints:

- `train.pkl`
- `valid.pkl`

Each pickle maps canonical SMILES to an RDKit `ExplicitBitVect`.

### `Feature_Morgan/by_task/<task>/`

Stores pickled dictionaries of Feature Morgan fingerprints:

- `train.pkl`
- `valid.pkl`

These are generated with `useFeatures=True`.

### `Morgan_similarity/by_task/<task>/`

Stores similarity lookup tables built from standard Morgan fingerprints:

- `train_similarity.pkl`
- `valid_similarity.pkl`

Meaning:

- `valid_similarity.pkl`: each valid query is compared against train references,
- `train_similarity.pkl`: each train query is compared against train references, excluding itself.

### `Feature_Morgan_similarity/by_task/<task>/`

Same structure as `Morgan_similarity/`, but for Feature Morgan fingerprints.

### `Label_maps/by_task/<task>/`

Stores canonicalized, safe label records:

- `train_labels.jsonl`
- `valid_labels.jsonl`

Each line keeps canonical SMILES, the final label `Y`, and bookkeeping fields such as source examples and row counts. Downstream code uses these files as the authoritative mapping from canonical SMILES to labels.

### `Label_conflicts/by_task/<task>/`

Stores excluded ambiguous examples:

- `train_excluded_conflicts.jsonl`
- `valid_excluded_conflicts.jsonl`

These records capture cases where the same canonical SMILES maps to multiple labels, either already in raw data or only after desalting/canonicalization.

### `KNN_pesudo_labels/by_task/<task>/`

This directory is not produced by `compute_fingerprints_and_similarities.py`; it is filled later by `train/KNN/extract_best_knn_labels.py`.

Files:

- `train_knn_labels.json`
- `valid_knn_labels.json`

These pseudo labels are then consumed by `DataPrepare/TDC_prepended/generate_knn_prompts.py`.

## Similarity File Format

Each similarity pickle is a dictionary keyed by canonical query SMILES:

```python
{
    query_smiles: {
        "query_label": 0 or 1,
        "label_0": [(similarity_score, reference_smiles), ...],
        "label_1": [(similarity_score, reference_smiles), ...],
    }
}
```

Both `label_0` and `label_1` lists are sorted from highest to lowest similarity.

This structure is designed so downstream KNN code can:

- preserve the true label of the query,
- recover all candidate references,
- combine Morgan and Feature Morgan similarity scores,
- perform ranked retrieval without recomputing RDKit fingerprints.

## Recommended Usage

### Step 1: Generate preprocessing artifacts

```bash
python DataPrepare/TDC_mol_fingerprints/compute_fingerprints_and_similarities.py
```

### Step 2: Sanity-check outputs

```bash
python DataPrepare/TDC_mol_fingerprints/verify_sims.py
python DataPrepare/TDC_mol_fingerprints/visualize.py DataPrepare/TDC_mol_fingerprints/Morgan_similarity/by_task/Carcinogens_Lagunin/valid_similarity.pkl 1
```

### Step 3: Run downstream KNN scripts

```bash
python train/KNN/eval_knn.py --tasks Carcinogens_Lagunin -k 3
python train/KNN/extract_best_knn_labels.py
```

### Step 4: Build KNN-augmented prompts

```bash
python DataPrepare/TDC_prepended/generate_knn_prompts.py --tasks Carcinogens_Lagunin --splits train valid --top-k 3
```

## Practical Notes

- The preprocessing script only handles `train` and `valid`; it does not generate artifacts for `test`.
- All downstream consumers assume the SMILES keys here are already desalted and canonicalized.
- The code uses multiprocessing, so runtime depends heavily on task size and available CPU cores.
- Several scripts use hard-coded task lists instead of full CLI configuration, so checking the bottom of each script before running is recommended.
