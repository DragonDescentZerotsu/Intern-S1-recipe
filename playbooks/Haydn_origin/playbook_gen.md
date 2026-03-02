# Playbook-generation directive (SMILES -> calibrated classifier)

Using your comprehensive set of literature search and analysis tools, create a detailed decision procedure (playbook) that a medicinal chemistry agent can follow to produce a calibrated likelihood of a small molecule (given as a SMILES string) has a desired property.

## 0) Agent task definition
- **Task / dataset:** Predicting whether a given SMILES string will cross the blood-brain barrier (BBB), evaluated on the TDC BBB_MARTINS dataset.
- **Positive class definition:** molecule crosses the BBB (`Y=1`)
- **base rate**: Not provided.
- **Output:** a **single integer 0–100** = calibrated probability of the positive class

## 1) What to output (hard requirements)
Output **only the playbook** as Markdown (no preamble) in the `answer` field. The playbook will be embedded verbatim in a larger agent prompt, so it must be **operational** (a decision procedure), not a literature essay. 

If your initial searches do not find enough relevant information, continue searching until you have sufficient data to design a robust playbook. Be sure to be clear with search intent so the summarization agent knows what to look for and what information to put in the final summary (e.g., "find experimental thresholds for logD or PSA that correlate with BBB permeability, include thresholds in output.").

### 1.1 Hard constraints
- **Length:** <= **3000 tokens**
- **Runtime limitation:** the downstream agent has **no access to papers/the web** at runtime. Use search/research only to *design* the rules, then bake the rules into the playbook.
- **Tools:** only refer to tools listed under **Available Tools For Molecular Property Analysis** below. Do **not** mention other tools or data sources.
- **No placeholders:** do not include ellipses (`...` or `…`) in tool-call examples, SMARTS, or reference SMILES lists.
- **SMARTS validity:** any SMARTS you include must be **RDKit-valid**.

### 1.2 Required playbook structure
Your playbook must include these sections (use these headings verbatim):
1) **Output Contract**
2) **Minimal Tool Recipe**
3) **High-Signal Motifs (SMARTS)**
4) **Scoring Rubric (0–100)**
5) **Guardrails & OOD Handling**

### 1.3 Calibration requirements (most important)
- Provide an explicit baseline prior `P0`:
  - If the **base rate** was provided in the header, set `P0 ≈ round(100 * base_rate)`.
  - Otherwise, use `P0 = 50` and clearly mark it as “unknown base rate”.
- Provide a deterministic mapping from tool outputs → probability:
  - additive points or a simple formula, then **clamp to [0,100]**
  - include an **uncertainty/OOD shrink** back toward **P0** (not always 50)
- Avoid undefined functions like `clamp()`/`sigmoid()` unless you define them and show how to compute them with `evaluate_arithmetic` (supports `exp`, `log`, `sqrt`, `sin`, `cos`, `round`, `min`, `max`).

### 1.4 Similarity/read-across rule
Only include a similarity/read-across section if you also include a **concrete** `reference_smiles` list inside the playbook (5–25 items, vetted). Otherwise omit similarity entirely (do not ask the runtime agent to "maintain a reference set").

# Available Tools For Molecular Property Analysis

Note on `standardize`: unless specified otherwise, `standardize=True` removes salts/counterions (keeps the largest organic fragment) and canonicalizes tautomers before analysis. `classify_ionization` is the exception and only removes salts/counterions.

## `standardize_smiles(smiles: str, remove_salts: bool = True, canonical_tautomer: bool = True, neutralize: bool = False) -> str`

Standardizes a SMILES with explicit control over each step. When enabled, removes salts/counterions (keeps largest organic fragment) and canonicalizes tautomers, then returns canonical isomeric SMILES. Optional neutralization uses RDKit's Uncharger and makes `formal_charge` descriptors meaningless. Raises on invalid or empty SMILES.

## `get_functional_groups(smiles: str, standardize: bool = False) -> str`

Runs AccFG and returns a plain-text, hierarchical functional-group tree. Use to spot alerting motifs (nitro/nitroso/epoxide/aziridine), ionizable groups, heterocycles, and general H-bonding functionality. Raises on invalid or empty SMILES.

## `compute_descriptors(smiles: str, descriptors: list[DescriptorName | str] | None = None, standardize: bool = False) -> dict`

Computes RDKit physicochemical descriptors. If `standardize=True`, removes salts/counterions and canonicalizes tautomers first. If `descriptors=None`, returns all groups. Descriptor names are case-insensitive strings or enum values; unknown names raise. Float outputs are rounded to 4 decimals.

| Descriptor Name       | Returned Fields                                                                      |
| --------------------- | ------------------------------------------------------------------------------------ |
| `masses`              | `average_mw`, `formula`                                                              |
| `atom_counts`         | `heavy_atoms`, `heteroatoms`, `nitrogen`, `oxygen`, `n_plus_o`                       |
| `surface_shape_props` | `tpsa`, `molar_refractivity`, `fraction_csp3`, `stereocenters`                       |
| `ring_counts`         | `total_rings`, `aromatic_rings`, `saturated_rings`                                   |
| `logp`                | Wildman-Crippen LogP (scalar)                                                        |
| `num_rotatable_bonds` | Lipinski rotatable bond count (scalar)                                               |
| `num_amide_bonds`     | Amide bond count (scalar)                                                            |
| `formal_charge`       | Net formal charge on the (standardized) input SMILES (scalar)                        |
| `qed`                 | Quantitative estimate of drug-likeness, 0-1 (scalar)                                 |
| `hydrogen_bonding`    | `hbd`, `hba`                                                                         |
| `lipinski_violations` | `mw_violation`, `logp_violation`, `hbd_violation`, `hba_violation`, `num_violations` |
| `esol`                | `log_s_esol`, `solubility_mg_per_ml`, `solubility_class`                             |

## `classify_ionization(smiles: str, ph: float = 7.4, standardize: bool = False) -> dict`

Uses Dimorphite-DL to enumerate protonation states at target pH (ph_min = ph_max = `ph`, precision = 0.5), summarizes charge-state distribution, and selects a representative variant. If `standardize=True`, only salts/counterions are removed (no tautomer canonicalization). If no variants are returned, the input is treated as neutral. Returns:
- `num_variants`: number of enumerated states
- `net_charges`: sorted unique net charges
- `charge_class_counts`: counts for `acid`, `base`, `zwitterion`, `neutral`
- `has_positive_states`, `has_negative_states`, `is_ambiguous`
- `representative`: `{smiles, net_charge, charge_class}` (mode net charge, tie-broken by lowest |net_charge|)

## `predict_pka(smiles: str, standardize: bool = False) -> pKaPredictionOutput`

Predicts pKa values for ionizable sites in a molecule.

Returns a `pKaPredictionOutput` with:
- `base_sites`: dict of atom-map-number (1-indexed) -> predicted base-site pKa
- `acid_sites`: dict of atom-map-number (1-indexed) -> predicted acid-site pKa
- `most_basic_pka`: max of `base_sites` (or `null` if none)
- `most_acidic_pka`: min of `acid_sites` (or `null` if none)
- `num_basic_sites`, `num_acidic_sites`
- `mapped_smiles`: SMILES with atom map numbers set (use to locate which atoms correspond to `base_sites`/`acid_sites`)

## `estimate_logd(smiles: str, ph: float = 7.4, standardize: bool = False) -> LogDEstimateOutput`

Estimates logD at a target pH using RDKit logP plus pKa-derived neutral-fraction heuristics:
- `logd`: estimated logD(pH) (approximate; not experimental)
- `logp`: RDKit Wildman-Crippen logP used in the estimate
- `fraction_neutral`: estimated fraction of unionized species at target pH
- `most_basic_pka`, `most_acidic_pka`, `num_basic_sites`, `num_acidic_sites`, `mapped_smiles`
- `warnings`: heuristic caveats (e.g., polyprotic molecules)

## `analyze_ring_systems(smiles: str, standardize: bool = False) -> dict`

Analyzes fused ring systems via shared bonds between SSSR rings. Returns:
- `ring_systems`: list of systems (sorted by `num_rings` desc, then `num_aromatic` desc), each with `num_rings`, `num_aromatic`, `all_aromatic`, `atom_count`, `ring_sizes` (sorted), `heteroatoms` (sorted unique)
- `num_ring_systems`, `largest_system_size`, `largest_aromatic_system`
- `has_pah_like`: `true` if any system has >=3 fused aromatic rings
- `largest_ring_size`, `has_macrocycle`, `num_macrocycles` (macrocycle = ring size >=12)
- `spiro_atoms`, `bridgehead_atoms`

## `match_substructure(smiles: str, patterns: dict[str, str], standardize: bool = False) -> dict[str, SubstructureMatchResult]`

SMARTS substructure screening. Takes a dict of pattern name -> SMARTS and returns a dict of name -> `{present, count}` where `count` is the number of RDKit match mappings (may include overlaps; no custom de-overlap). Raises on invalid SMARTS or SMILES. Do not pass ellipses (`...`) or abbreviated SMILES.

Example:
```
match_substructure("O=[N+]([O-])c1ccc(N)cc1", {
  "aromatic_nitro": "[$(N(=O)~[O]),$([N+](=O)[O-])]c",
  "aromatic_amine": "[NH2]c",
  "epoxide": "C1OC1"
})
-> {
  "aromatic_nitro": {"present": true, "count": 1},
  "aromatic_amine": {"present": true, "count": 1},
  "epoxide": {"present": false, "count": 0}
}
```

## `compute_similarity(smiles: str, reference_smiles: list[str], fingerprint: FingerprintType = FingerprintType.MORGAN, standardize: bool = False) -> dict`

Computes Tanimoto similarity between the query and each reference with RDKit fingerprints. `fingerprint` accepts a case-insensitive string or enum; valid values: `morgan`, `rdkit`, `maccs`, `atom_pair`, `topological_torsion`. Morgan uses radius=2, 2048 bits, with chirality; RDKit/atom-pair/torsion use 2048 bits; MACCS uses its built-in size. Returns `fingerprint` plus `similarities` as a list of `{reference_smiles, similarity}` (0-1, rounded to 4 decimals), sorted descending. Raises on invalid SMILES or fingerprint type.

## `find_mcs(smiles: str, reference_smiles: list[str], complete_rings_only: bool = True, ring_matches_ring_only: bool = True, standardize: bool = False) -> dict`

Maximum common substructure across query + references (RDKit MCS, 5s timeout). Returns `smarts`, `smiles` (string or `null` if not convertible), `num_atoms`, `num_bonds`, `canceled`, `query_coverage` (fraction of query heavy atoms), and `ref_coverages` (fractions of each reference heavy atoms, same order as inputs). Float outputs are rounded to 4 decimals.

## `score_structural_alerts(smiles: str, alert_library: AlertLibrary = AlertLibrary.ALL, standardize: bool = False) -> dict`

Runs RDKit FilterCatalog alerts. `alert_library` accepts a case-insensitive string or enum; valid values: `all`, `pains`, `pains_a`, `pains_b`, `pains_c`, `brenk`, `nih`, `zinc`, `chembl`, `chembl_bms`, `chembl_lint`, `chembl_mlsmr`. Returns `library`, `count`, and `alerts`, where each alert includes `description`, `filter_set`, `reference`, and `scope` (may be `null`). Raises on invalid SMILES or library.

## `extract_pharmacophore_features(smiles: str, standardize: bool = False) -> dict`

Extracts RDKit BaseFeatures pharmacophore-like features. Returns `feature_counts` by family and a `features` list of `{family, type, atom_ids}`. No 3D coordinates are produced. Raises on invalid SMILES.

## `get_murcko_scaffold(smiles: str, standardize: bool = False, generic: bool = False) -> dict`

Extracts the Bemis-Murcko scaffold. Returns `scaffold_smiles` (empty string if none), `generic_scaffold_smiles` (`null` when `generic=False`, otherwise a string, possibly empty), `num_scaffold_atoms`, `num_scaffold_rings`, and `scaffold_fraction` (scaffold heavy atoms / molecule heavy atoms).

## `evaluate_arithmetic(expression: str) -> float`

Evaluate an arithmetic expression; use for scoring rubrics and threshold checks.
