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