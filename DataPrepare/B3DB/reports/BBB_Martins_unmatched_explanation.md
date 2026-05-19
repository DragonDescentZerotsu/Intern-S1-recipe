# Why Some TDC BBB_Martins Molecules Do Not Strictly Match B3DB

## Files downloaded/read

- `Meng_Xi_Huang_Ayers_2021_B3DB_Scientific_Data.pdf`: B3DB Scientific Data paper.
- `Meng_Xi_Huang_Ayers_2021_B3DB_Scientific_Data.txt`: text extracted from the PDF for local review.
- `B3DB_raw_data_summary.tsv`: B3DB raw-source summary from the official GitHub repository.
- `B3DB_raw_R1_Martins_data_formatted_done.xls`: B3DB raw formatted Martins source file.
- `tdc_228_strict_unmatched_vs_b3db_raw_R1.csv`: strict-unmatched TDC rows checked against B3DB raw R1/Martins.
- `tdc_144_unmatched_vs_b3db_raw_R1.csv`: relaxed-unmatched TDC rows checked against B3DB raw R1/Martins.
- `tdc_u144_inchikey_check.csv`: the 144 relaxed-unmatched rows checked against final B3DB by RDKit InChIKey.
- `tdc_123_raw_R1_but_absent_from_final_B3DB.csv`: rows found in raw R1 but absent from final B3DB by InChIKey.
- `B3DB_data_curation_README.md`: official B3DB data-curation README.

## Key conclusion

Yes, Martins et al. is one of B3DB's data sources. In B3DB it is raw source `R1`, and the paper table lists it as 2053 categorical records. However, B3DB is not a direct copy of Martins/TDC BBB_Martins. It is a curated, merged, molecule-level database built from 50 sources. The final B3DB table keeps only curated molecular records after SMILES repair, PubChem/PUG-REST canonical/isomeric replacement, salt/solvent stripping, charge neutralization, deduplication, cross-source label conflict handling, and categorical grouping.

That is enough to explain why strict RDKit isomeric canonical matching finds 228 TDC rows unmatched:

- B3DB final classification table has 7982 rows after adding the 2025 external set.
- Only 1524 final B3DB rows cite `R1`/Martins in the `reference` field, even though the raw Martins source has 2053 records. Many Martins records were merged with other sources, transformed to a different curated representation, or removed during curation.
- TDC `BBB_Martins` has 2030 rows in this project.
- Our strict matching uses RDKit canonical isomeric SMILES, plus a second pass after RDKit largest-fragment salt removal.

## What "adding PubChem isomeric SMILES" means

A canonical SMILES is a deterministic string for a molecular graph. An isomeric SMILES additionally encodes stereochemistry where known: chiral centers such as `@`/`@@`, and double-bond geometry such as `/` and `\`.

B3DB says it used PubChem/PUG-REST to upgrade generic SMILES to isomeric SMILES wherever possible. In practice, a source record may have a non-stereochemical SMILES, while B3DB's final row may have a stereochemical PubChem version of the same compound. RDKit strict isomeric canonical matching treats these as different keys, because "unknown stereochemistry" and "specified stereochemistry" are not the same molecular representation.

That is why some TDC rows look unmatched under strict isomeric SMILES but reappear when stereochemistry is ignored.

## Breakdown of the 228 strict-unmatched TDC rows

From `tdc_original_all_rows_audit.csv` plus an extra non-isomeric relaxed check:

- 228 rows are strict-unmatched.
- They represent 223 unique exact canonical SMILES and 222 unique salt-removed canonical SMILES.
- 24 rows have multiple fragments and changed after salt removal.
- 8 rows are part of TDC internal exact-label conflicts.
- 8 rows are part of TDC internal salt-removed-label conflicts.
- 74 of the 228 rows match B3DB if stereochemistry is ignored:
  - 72 by non-isomeric exact canonical SMILES.
  - 2 by non-isomeric salt-removed canonical SMILES.
  - 59 of these 74 have labels agreeing with B3DB.
  - 15 have labels disagreeing with B3DB.
- 10 more rows hit B3DB non-isomeric keys whose B3DB labels are ambiguous under that relaxed key.
- 144 rows remain unmatched even after non-isomeric salt-removed matching.

## Where did the 144 remaining rows go?

They did not disappear before B3DB ingested Martins. After parsing the B3DB raw R1/Martins `.xls`, all 144 rows were found in raw R1 under exact/salt/non-isomeric matching. The same is true for all 228 strict-unmatched rows.

Against the final B3DB table:

- 21 of the 144 rows match final B3DB by RDKit InChIKey even though they did not match by non-isomeric canonical SMILES. These are likely tautomer, charge, neutralization, or other standardization differences introduced by B3DB's ChEMBL/PubChem cleaning.
- 123 of the 144 rows do not match final B3DB even by RDKit InChIKey. These are present in raw Martins/R1 but absent from the final B3DB classification molecule set under the identifiers we can reproduce locally.

For those 123 final-absent rows:

- 87 are from TDC train, 13 from valid, 23 from test.
- Labels are 69 negative and 54 positive.
- 22 are multi-fragment/salt records under the original TDC representation.
- 2 are involved in TDC internal label conflicts.
- 1 contains atom `Z > 20` in the original TDC representation.
- 16 contain atoms in the metal/counterion-like set used by B3DB's cleaning script, mostly because of salts/counterions.

The B3DB grouping script shows two relevant deletion modes after cleaning and InChI-level merging:

- `dropped_group1`: threshold `-1` categorical sources with inconsistent labels.
- `dropped_group2`: no-threshold categorical sources where voting does not resolve the conflict.

Since Martins/R1 is a threshold `-1` categorical source, raw R1 molecules can be removed if, after B3DB's structure standardization and InChI merging, they conflict with another threshold `-1` categorical record. They can also be removed earlier by structure cleaning/standardization filters, although the exact intermediate B3DB workbook that records each dropped row is not shipped as a final data file in the repository.

The practical interpretation is:

1. Some strict-unmatched rows are representation mismatches caused by B3DB adding stereochemistry through PubChem/PUG-REST. These are not necessarily absent from final B3DB.
2. Some rows are salts/mixtures/alternate forms where B3DB's ChEMBL/OpenEye/PubChem standardization may not equal our RDKit largest-fragment-only standardization.
3. For the 144 relaxed-unmatched rows: all are in B3DB raw Martins/R1; 21 survive final B3DB under InChIKey but not SMILES matching; 123 appear to have been dropped or transformed beyond reproducible InChIKey matching during B3DB cleaning/curation.
4. Since the requested audit defines B3DB final data as the gold standard and asks for strict matching, the 228 rows should remain "unverified by B3DB" under that strict policy rather than automatically counted as correct.

## B3DB curation details relevant to this

From the paper:

- Raw data came from 50 publications/open datasets.
- Raw records were converted to a standard template with compound name, SMILES, PubChem CID, logBB, categorical BBB label, InChI, threshold, and source reference.
- Invalid SMILES were fixed where possible; missing compound names/SMILES/CIDs were retrieved through PubChem.
- SMILES were upgraded to isomeric SMILES by PUG-REST where possible; otherwise canonical PubChem SMILES were used.
- ChEMBL Structure Pipeline was used to strip salts/solvents and neutralize charge.
- The categorical curation groups were:
  - Group A: molecules with numerical logBB, converted to BBB+/BBB- using logBB threshold -1.
  - Group B: molecules from sources using threshold -1 where all sources agree.
  - Group C: molecules where sources agree but threshold is not reported.
  - Group D: molecules with conflicting labels where the majority label is used; 45 equal-frequency conflicts were discarded.

This explains the discrepancy: B3DB may include Martins as a source but its final molecule-level labels are not equal to TDC's raw-ish BBB_Martins rows.
