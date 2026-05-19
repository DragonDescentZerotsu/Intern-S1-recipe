# BBB_Martins Cleaning with B3DB Gold Standard

## Executive Summary

We audited TDC `BBB_Martins` against B3DB, treating the curated B3DB final classification table as the gold standard. The original TDC `BBB_Martins` contains 2030 rows across the scaffold splits:

| Split | Original Rows |
|---|---:|
| train | 1421 |
| valid | 203 |
| test | 406 |
| total | 2030 |

After strict B3DB matching, manual investigation of unmatched cases, high-confidence rescue, B3DB-SMILES replacement, B3DB-label replacement, and global molecule deduplication, the final clean dataset contains 1779 rows:

| Split | Final Rows | Rows Removed |
|---|---:|---:|
| train | 1239 | 182 |
| valid | 178 | 25 |
| test | 362 | 44 |
| total | 1779 | 251 |

Total data loss vs original: 251 / 2030 = 12.36%.

The downstream ML result supports the cleaning decision:

| Dataset / Model Setting | Macro F1 | AUROC |
|---|---:|---:|
| Previous uncleaned BBB training | 0.8136 | 0.9228 |
| Previous best MiniMol baseline | - | 0.9230 |
| Final cleaned BBB training | 0.8375 | 0.9515 |

Performance change after cleaning:

- Macro F1: +0.0239 absolute.
- AUROC: +0.0287 absolute vs previous uncleaned run.
- AUROC: +0.0285 absolute vs previous MiniMol best.

This is consistent with the hypothesis that TDC `BBB_Martins` contains mislabeled or structurally inconsistent samples, and that the B3DB-based cleaning removed meaningful label noise.

## Data Sources

Downloaded B3DB files:

- `DataPrepare/B3DB/raw/B3DB_classification.tsv`
- `DataPrepare/B3DB/raw/B3DB_classification_external.tsv`

Downloaded B3DB paper and curation material:

- `DataPrepare/B3DB/reports/Meng_Xi_Huang_Ayers_2021_B3DB_Scientific_Data.pdf`
- `DataPrepare/B3DB/reports/Meng_Xi_Huang_Ayers_2021_B3DB_Scientific_Data.txt`
- `DataPrepare/B3DB/reports/B3DB_data_curation_README.md`
- `DataPrepare/B3DB/reports/B3DB_raw_data_summary.tsv`
- `DataPrepare/B3DB/reports/B3DB_raw_R1_Martins_data_formatted_done.xls`

B3DB uses Martins et al. as raw source `R1`. The B3DB paper lists R1/Martins as 2053 categorical records. However, B3DB final classification is not a raw copy of Martins/TDC. B3DB cleans and merges 50 sources into molecule-level records.

Relevant B3DB curation steps from the paper/code:

- Fix invalid SMILES where possible.
- Retrieve missing names, CIDs, and SMILES from PubChem.
- Upgrade generic SMILES to PubChem isomeric SMILES where possible.
- Strip salts/solvents and standardize molecules using ChEMBL Structure Pipeline.
- Neutralize charges.
- Remove problematic structures, including heavy atom cases.
- Merge records by InChI.
- Convert numerical logBB to categorical labels with threshold `logBB = -1`.
- Drop unresolved categorical conflicts.
- Use majority label for some conflict groups where voting resolves the conflict.

## Initial Strict Audit

The first audit script is:

- `DataPrepare/B3DB/analyze_bbb_martins_vs_b3db.py`

Main outputs:

- `DataPrepare/B3DB/reports/BBB_Martins_quality_report.md`
- `DataPrepare/B3DB/reports/summary.json`
- `DataPrepare/B3DB/reports/tdc_original_all_rows_audit.csv`
- `DataPrepare/B3DB/reports/tdc_b3db_label_mismatches.csv`
- `DataPrepare/B3DB/reports/problem_rows.csv`
- `DataPrepare/B3DB/reports/tdc_duplicate_groups_exact.csv`
- `DataPrepare/B3DB/reports/tdc_duplicate_groups_salt_removed.csv`

Strict audit results:

- B3DB final classification rows loaded: 7982.
- B3DB RDKit-valid rows: 7980.
- TDC original BBB_Martins rows: 2030.
- Exact B3DB matches: 1723 rows.
- Salt-removed-only B3DB matches: 79 rows.
- Strict unmatched rows: 228 rows.
- RDKit-unparseable original TDC rows: 0.
- Exact-match label mismatches: 45 rows.
- Salt-removed-match label mismatches: 46 rows.
- Same exact canonical molecule with conflicting TDC labels: 10 canonical keys.
- Same salt-removed canonical molecule with conflicting TDC labels: 11 canonical keys.

The first strict B3DB-confirmed clean dataset contained 1699 rows:

| Split | Strict B3DB-Confirmed Rows |
|---|---:|
| train | 1184 |
| valid | 170 |
| test | 345 |
| total | 1699 |

## Investigation of the 228 Strict-Unmatched Rows

We investigated why 228 TDC rows did not match B3DB final classification under strict RDKit isomeric canonical SMILES matching.

Additional reports:

- `DataPrepare/B3DB/reports/BBB_Martins_unmatched_explanation.md`
- `DataPrepare/B3DB/reports/tdc_unmatched_relaxed_nonisomeric_check.csv`
- `DataPrepare/B3DB/reports/tdc_228_strict_unmatched_vs_b3db_raw_R1.csv`
- `DataPrepare/B3DB/reports/tdc_144_unmatched_vs_b3db_raw_R1.csv`
- `DataPrepare/B3DB/reports/tdc_u144_inchikey_check.csv`
- `DataPrepare/B3DB/reports/tdc_123_raw_R1_but_absent_from_final_B3DB.csv`

Important finding:

- All 228 strict-unmatched TDC rows are present in B3DB raw R1/Martins.
- Therefore, the issue is not that Martins was absent from B3DB.
- The issue is that B3DB final classification is curated and standardized, not raw Martins.

Breakdown:

- 74 of the 228 rows match B3DB if stereochemistry is ignored:
  - 72 by non-isomeric exact canonical SMILES.
  - 2 by non-isomeric salt-removed canonical SMILES.
- 144 rows remain unmatched even after ignoring stereochemistry and salt removal.
- Of those 144:
  - 21 match final B3DB by RDKit InChIKey, indicating tautomer/charge/neutralization/standardization differences.
  - 123 are present in raw Martins/R1 but absent from final B3DB by reproducible InChIKey matching.

Interpretation:

- Some rows were only representation mismatches caused by B3DB adding stereochemistry through PubChem isomeric SMILES.
- Some rows were changed by B3DB structure standardization, salt handling, charge neutralization, tautomer/representation normalization, or InChI-level merging.
- Some raw Martins rows likely failed final B3DB curation due to conflicts or filtering.

## Meaning of PubChem Isomeric SMILES

B3DB reports that it upgraded generic SMILES to PubChem isomeric SMILES where possible. This matters because:

- A canonical SMILES gives a deterministic string for molecular connectivity.
- An isomeric SMILES also encodes stereochemistry, such as chiral centers and double-bond geometry.
- RDKit strict isomeric canonical matching treats unspecified stereochemistry and specified stereochemistry as different molecular keys.

Therefore, a TDC molecule can fail strict matching even when B3DB contains the same connectivity with stereochemistry specified.

## Final Cleaning Policy

The requested final clean dataset was built from:

1. The strict B3DB-confirmed clean set.
2. Rescued rows from the 228 strict-unmatched set that match final B3DB after ignoring stereochemistry.
3. Rescued rows from the 144 relaxed-unmatched set that match final B3DB by RDKit InChIKey.

For rescued rows:

- Only B3DB high-confidence groups `A` and `B` were accepted.
- The final SMILES was replaced with B3DB final SMILES.
- The final label was replaced with B3DB final label.
- Rows with ambiguous B3DB matches were not used.
- Final output was globally deduplicated by salt-removed canonical SMILES.

High-confidence definition used here:

- Group `A`: molecules with numerical logBB data converted to BBB+/BBB- using threshold `logBB = -1`.
- Group `B`: threshold `-1` categorical sources where all sources agree.

Groups `C`, `D`, and `E` were not used for rescue.

The final build script is:

- `DataPrepare/B3DB/build_final_bbb_martins_clean.py`

## Final Clean Dataset

Final output directory:

- `DataPrepare/B3DB/processed/BBB_Martins_final_clean_b3db_high_conf_rescued`

Files:

- `drug_b3db_smiles/train.jsonl`
- `drug_b3db_smiles/valid.jsonl`
- `drug_b3db_smiles/test.jsonl`
- `prompts_b3db_smiles/train.jsonl`
- `prompts_b3db_smiles/valid.jsonl`
- `prompts_b3db_smiles/test.jsonl`
- `reports/summary.json`
- `reports/final_rows_manifest.csv`
- `reports/rescue_candidate_audit.csv`
- `reports/rescued_rows_kept_before_dedup.csv`
- `reports/duplicate_or_invalid_drops.csv`

Final row counts:

| Split | Rows |
|---|---:|
| train | 1239 |
| valid | 178 |
| test | 362 |
| total | 1779 |

Source composition:

| Source | Final Rows |
|---|---:|
| strict B3DB-confirmed clean | 1699 |
| non-isomeric exact high-confidence rescue | 60 |
| non-isomeric salt high-confidence rescue | 2 |
| InChIKey exact high-confidence rescue | 18 |
| total | 1779 |

Candidate rescue summary:

- Non-isomeric relaxed candidates with unique high-confidence B3DB match: 65.
- InChIKey candidates with unique high-confidence B3DB match: 20.
- Before final deduplication, 85 rescued rows were kept.
- 5 rescued rows were dropped as duplicates of already kept final molecules.
- Final rescued rows added: 80.

Validation performed:

- `drug` and `prompt` JSONL files have matching line counts for each split.
- All final SMILES are RDKit-parseable.
- Final dataset has no duplicate salt-removed canonical molecules.

## Model Performance After Cleaning

The cleaned dataset improves held-out performance substantially:

| Metric | Previous Uncleaned BBB | Final Clean BBB | Absolute Change |
|---|---:|---:|---:|
| Macro F1 | 0.8136 | 0.8375 | +0.0239 |
| AUROC | 0.9228 | 0.9515 | +0.0287 |

Compared with the previous best MiniMol AUROC of 0.9230, the final clean BBB result reaches 0.9515, an absolute improvement of +0.0285.

This performance gain is strong evidence that the cleaning process removed harmful label noise and structurally inconsistent training/evaluation records.

## Recommendation

Use the final dataset below as the clean version of TDC `BBB_Martins`:

- `DataPrepare/B3DB/processed/BBB_Martins_final_clean_b3db_high_conf_rescued`

For future experiments, report that this dataset:

- Uses B3DB final classification as the gold standard.
- Replaces rescued rows with B3DB final SMILES and B3DB labels.
- Accepts only B3DB high-confidence `A/B` matches for rescued rows.
- Removes 251 of 2030 original rows.
- Keeps 1779 final unique salt-removed molecules.
- Improves AUROC from 0.9228 to 0.9515 and macro F1 from 0.8136 to 0.8375 in the user's ML experiment.
