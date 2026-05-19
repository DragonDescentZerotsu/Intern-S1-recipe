# BBB_Martins vs B3DB Quality Audit

- Generated at: 2026-05-01T20:21:14.412489+00:00
- B3DB source files: `DataPrepare/B3DB/raw/B3DB_classification.tsv`, `DataPrepare/B3DB/raw/B3DB_classification_external.tsv`
- B3DB label mapping: `BBB+ -> 1` (crosses BBB), `BBB- -> 0` (does not cross BBB).
- Matching policy: RDKit canonical isomeric SMILES exact match first; RDKit LargestFragmentChooser salt-removed canonical match second.

## Top-line counts

- TDC original BBB_Martins rows: 2030; split counts: {'train': 1421, 'valid': 203, 'test': 406}
- TDC no-conflict salt-removed BBB_Martins rows already present: 1956; split counts: {'train': 1369, 'valid': 195, 'test': 392}
- B3DB classification rows loaded: 7982 (7980 RDKit-valid, 2 invalid)
- Original TDC rows with exact B3DB match: 1723
- Original TDC rows with salt-removed-only B3DB match: 79
- Original TDC rows unmatched to B3DB: 228
- Hard-problem original TDC rows removed by the clean filter: 61
- Clean filtered rows after hard-problem removal and salt-removed dedup: 1917; split counts: {'train': 1339, 'valid': 193, 'test': 385}
- B3DB-confirmed clean rows after the same dedup: 1699; split counts: {'train': 1184, 'valid': 170, 'test': 345}

## Label quality against B3DB

- Exact-match label mismatches: 45 rows.
- Salt-removed-match label mismatches: 46 rows.
- TDC rows whose label is confirmed correct by B3DB exact match: 1672.
- TDC rows whose label is confirmed correct only after salt removal: 77.

## Internal TDC issues

- RDKit-unparseable original TDC rows: 0.
- Same original canonical molecule with conflicting labels: 10 canonical keys.
- Same salt-removed canonical molecule with conflicting labels: 11 canonical keys.
- Salt removal collapses multiple original molecules into the same canonical molecule: 9 salt-removed keys.
- B3DB exact canonical keys with conflicting labels: 2.
- B3DB salt-removed canonical keys with conflicting labels: 2.

## Output files

- `reports/problem_rows.csv`: all original TDC rows with hard-removal flags.
- `reports/tdc_b3db_label_mismatches.csv`: rows whose TDC label disagrees with B3DB.
- `reports/tdc_duplicate_groups_exact.csv`: duplicate/conflicting groups by original canonical SMILES.
- `reports/tdc_duplicate_groups_salt_removed.csv`: duplicate/conflicting groups by salt-removed canonical SMILES.
- `reports/b3db_conflict_groups_exact.csv` and `reports/b3db_conflict_groups_salt_removed.csv`: B3DB-internal conflicting canonical keys.
- `processed/BBB_Martins_filtered_clean/drug_salt_removed/{train,valid,test}.jsonl`: clean compatible `drug`/`Y` JSONL.
- `processed/BBB_Martins_filtered_clean/prompts_salt_removed/{train,valid,test}.jsonl`: clean prompt JSONL with SMILES replaced by salt-removed canonical SMILES.
- `processed/BBB_Martins_b3db_confirmed_clean/...`: stricter subset containing only B3DB-matched agreeing molecules.
