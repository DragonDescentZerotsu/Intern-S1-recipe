# Threshold Research Brief: BBB_Martins

- Experiment: `fg_top_level_plus_rdkit_pka_easy_to_NLP_Lv1_traincv5_n40_eval0_jobs16`
- Feature set: `fg_top_level+rdkit_descriptors_and_pka_easy_to_NLP_Lv1`
- Feature universe source: `results_surviving`
- Included reasoning files: 0
- Surviving features from task results: 666
- Playbook candidate features: 132
- Included source families: pka, rdkit
- Excluded source families: fg_top_level
- Class A: does not cross the BBB
- Class B: crosses the BBB

## Research Scope

- Prefer the task's surviving feature universe from result artifacts as the stable source of candidate features.
- Use reasoning JSON observations as priority signals, not as the sole feature-discovery source.
- Focus literature threshold collection on the features below.
- Do not search literature thresholds for functional-group indicators/counts from `fg_top_level`.
- Use threshold examples from reasoning data only as model-side context, not as evidence.

## Threshold-Research Features

- `rdkit_pka__rdkit__TPSA`: topological polar surface area (source family: rdkit; raw name: TPSA; description: topological polar surface area of the molecule)
- `rdkit_pka__rdkit__RingCount`: ring count (source family: rdkit; raw name: RingCount; description: total number of rings)
- `rdkit_pka__rdkit__NumSaturatedRings`: saturated ring count (source family: rdkit; raw name: NumSaturatedRings; description: number of saturated rings)
- `rdkit_pka__rdkit__NumSaturatedHeterocycles`: saturated heterocycle count (source family: rdkit; raw name: NumSaturatedHeterocycles; description: number of saturated heterocyclic rings)
- `rdkit_pka__rdkit__NumSaturatedCarbocycles`: saturated carbocycle count (source family: rdkit; raw name: NumSaturatedCarbocycles; description: number of saturated carbocyclic rings)
- `rdkit_pka__rdkit__NumRotatableBonds`: rotatable-bond count (source family: rdkit; raw name: NumRotatableBonds; description: number of rotatable bonds)
- `rdkit_pka__rdkit__NumHeteroatoms`: heteroatom count (source family: rdkit; raw name: NumHeteroatoms; description: number of heteroatoms, such as N, O, or S)
- `rdkit_pka__rdkit__NumHDonors`: hydrogen-bond donor count (source family: rdkit; raw name: NumHDonors; description: number of hydrogen-bond donors)
- `rdkit_pka__rdkit__NumHAcceptors`: hydrogen-bond acceptor count (source family: rdkit; raw name: NumHAcceptors; description: number of hydrogen-bond acceptors)
- `rdkit_pka__rdkit__NumAromaticRings`: aromatic ring count (source family: rdkit; raw name: NumAromaticRings; description: number of aromatic rings)
- `rdkit_pka__rdkit__NumAromaticHeterocycles`: aromatic heterocycle count (source family: rdkit; raw name: NumAromaticHeterocycles; description: number of aromatic heterocyclic rings)
- `rdkit_pka__rdkit__NumAromaticCarbocycles`: aromatic carbocycle count (source family: rdkit; raw name: NumAromaticCarbocycles; description: number of aromatic carbocyclic rings)
- `rdkit_pka__rdkit__NumAliphaticRings`: aliphatic ring count (source family: rdkit; raw name: NumAliphaticRings; description: number of aliphatic rings)
- `rdkit_pka__rdkit__NumAliphaticHeterocycles`: aliphatic heterocycle count (source family: rdkit; raw name: NumAliphaticHeterocycles; description: number of aliphatic heterocyclic rings)
- `rdkit_pka__rdkit__NumAliphaticCarbocycles`: aliphatic carbocycle count (source family: rdkit; raw name: NumAliphaticCarbocycles; description: number of aliphatic carbocyclic rings)
- `rdkit_pka__rdkit__NOCount`: nitrogen/oxygen atom count (source family: rdkit; raw name: NOCount; description: number of nitrogen and oxygen atoms)
- `rdkit_pka__rdkit__NHOHCount`: NH/OH group count (source family: rdkit; raw name: NHOHCount; description: number of NH or OH groups)
- `rdkit_pka__rdkit__MolWt`: molecular weight (source family: rdkit; raw name: MolWt; description: molecular weight)
- `rdkit_pka__rdkit__MolLogP`: estimated logP (source family: rdkit; raw name: MolLogP; description: RDKit-estimated octanol/water partition coefficient (logP))
- `rdkit_pka__rdkit__LabuteASA`: Labute surface area (source family: rdkit; raw name: LabuteASA; description: Labute approximate surface area)
- `rdkit_pka__rdkit__HeavyAtomMolWt`: heavy-atom molecular weight (source family: rdkit; raw name: HeavyAtomMolWt; description: molecular weight contributed by heavy atoms)
- `rdkit_pka__rdkit__HeavyAtomCount`: heavy-atom count (source family: rdkit; raw name: HeavyAtomCount; description: number of non-hydrogen atoms)
- `rdkit_pka__rdkit__FractionCSP3`: fraction of sp3 carbons (source family: rdkit; raw name: FractionCSP3; description: fraction of carbon atoms that are sp3 hybridized)
- `rdkit_pka__rdkit__ExactMolWt`: exact molecular weight (source family: rdkit; raw name: ExactMolWt; description: exact isotopic molecular weight)
- `rdkit_pka__pka__num_ionizable_sites`: number of ionizable sites (source family: pka; raw name: num_ionizable_sites; description: total number of acidic and basic ionizable sites)
- `rdkit_pka__pka__num_basic_sites`: number of basic sites (source family: pka; raw name: num_basic_sites; description: number of basic ionizable sites in the molecule)
- `rdkit_pka__pka__num_acidic_sites`: number of acidic sites (source family: pka; raw name: num_acidic_sites; description: number of acidic ionizable sites in the molecule)
- `rdkit_pka__pka__logp_wildman_crippen`: estimated logP (source family: pka; raw name: logp_wildman_crippen; description: Wildman-Crippen estimated logP)
- `rdkit_pka__pka__logd_ph`: logD pH setting (source family: pka; raw name: logd_ph; description: pH value used when estimating logD)
- `rdkit_pka__pka__logd_estimate`: estimated logD (source family: pka; raw name: logd_estimate; description: estimated logD at the configured pH)
- `rdkit_pka__pka__fraction_neutral`: neutral fraction (source family: pka; raw name: fraction_neutral; description: estimated fraction of the molecule that is neutral at the configured pH)
- `rdkit_pka__pka__base_site_pka_sum`: sum basic site pKa (source family: pka; raw name: base_site_pka_sum; description: sum pKa across the basic ionizable sites)
- `rdkit_pka__pka__acid_site_pka_sum`: sum acidic site pKa (source family: pka; raw name: acid_site_pka_sum; description: sum pKa across the acidic ionizable sites)

## Qualitative-Rewrite Features

- `rdkit_pka__rdkit__qed`: QED drug-likeness (source family: rdkit; raw name: qed; description: quantitative estimate of drug-likeness)
- `rdkit_pka__rdkit__fr_urea`: urea fragment count (source family: rdkit; raw name: fr_urea; description: count of RDKit-recognized urea fragments)
- `rdkit_pka__rdkit__fr_unbrch_alkane`: unbrch alkane fragment count (source family: rdkit; raw name: fr_unbrch_alkane; description: count of RDKit-recognized unbrch alkane fragments)
- `rdkit_pka__rdkit__fr_thiophene`: thiophene fragment count (source family: rdkit; raw name: fr_thiophene; description: count of RDKit-recognized thiophene fragments)
- `rdkit_pka__rdkit__fr_thiocyan`: thiocyan fragment count (source family: rdkit; raw name: fr_thiocyan; description: count of RDKit-recognized thiocyan fragments)
- `rdkit_pka__rdkit__fr_thiazole`: thiazole fragment count (source family: rdkit; raw name: fr_thiazole; description: count of RDKit-recognized thiazole fragments)
- `rdkit_pka__rdkit__fr_tetrazole`: tetrazole fragment count (source family: rdkit; raw name: fr_tetrazole; description: count of RDKit-recognized tetrazole fragments)
- `rdkit_pka__rdkit__fr_term_acetylene`: term acetylene fragment count (source family: rdkit; raw name: fr_term_acetylene; description: count of RDKit-recognized term acetylene fragments)
- `rdkit_pka__rdkit__fr_sulfone`: sulfone fragment count (source family: rdkit; raw name: fr_sulfone; description: count of RDKit-recognized sulfone fragments)
- `rdkit_pka__rdkit__fr_sulfonamd`: sulfonamd fragment count (source family: rdkit; raw name: fr_sulfonamd; description: count of RDKit-recognized sulfonamd fragments)
- `rdkit_pka__rdkit__fr_sulfide`: sulfide fragment count (source family: rdkit; raw name: fr_sulfide; description: count of RDKit-recognized sulfide fragments)
- `rdkit_pka__rdkit__fr_quatN`: quaternary nitrogen count (source family: rdkit; raw name: fr_quatN; description: count of quaternary ammonium or quaternary nitrogen centers)
- `rdkit_pka__rdkit__fr_pyridine`: pyridine fragment count (source family: rdkit; raw name: fr_pyridine; description: count of RDKit-recognized pyridine fragments)
- `rdkit_pka__rdkit__fr_prisulfonamd`: prisulfonamd fragment count (source family: rdkit; raw name: fr_prisulfonamd; description: count of RDKit-recognized prisulfonamd fragments)
- `rdkit_pka__rdkit__fr_priamide`: priamide fragment count (source family: rdkit; raw name: fr_priamide; description: count of RDKit-recognized priamide fragments)
- `rdkit_pka__rdkit__fr_piperzine`: piperzine fragment count (source family: rdkit; raw name: fr_piperzine; description: count of RDKit-recognized piperzine fragments)
- `rdkit_pka__rdkit__fr_piperdine`: piperdine fragment count (source family: rdkit; raw name: fr_piperdine; description: count of RDKit-recognized piperdine fragments)
- `rdkit_pka__rdkit__fr_phos_ester`: phos ester fragment count (source family: rdkit; raw name: fr_phos_ester; description: count of RDKit-recognized phos ester fragments)
- `rdkit_pka__rdkit__fr_phos_acid`: phos acid fragment count (source family: rdkit; raw name: fr_phos_acid; description: count of RDKit-recognized phos acid fragments)
- `rdkit_pka__rdkit__fr_phenol_noOrthoHbond`: phenol noOrthoHbond fragment count (source family: rdkit; raw name: fr_phenol_noOrthoHbond; description: count of RDKit-recognized phenol noOrthoHbond fragments)
- `rdkit_pka__rdkit__fr_phenol`: phenol fragment count (source family: rdkit; raw name: fr_phenol; description: count of RDKit-recognized phenol fragments)
- `rdkit_pka__rdkit__fr_para_hydroxylation`: para hydroxylation fragment count (source family: rdkit; raw name: fr_para_hydroxylation; description: count of RDKit-recognized para hydroxylation fragments)
- `rdkit_pka__rdkit__fr_oxime`: oxime fragment count (source family: rdkit; raw name: fr_oxime; description: count of RDKit-recognized oxime fragments)
- `rdkit_pka__rdkit__fr_oxazole`: oxazole fragment count (source family: rdkit; raw name: fr_oxazole; description: count of RDKit-recognized oxazole fragments)
- `rdkit_pka__rdkit__fr_nitroso`: nitroso fragment count (source family: rdkit; raw name: fr_nitroso; description: count of RDKit-recognized nitroso fragments)
- `rdkit_pka__rdkit__fr_nitro_arom_nonortho`: nitro arom nonortho fragment count (source family: rdkit; raw name: fr_nitro_arom_nonortho; description: count of RDKit-recognized nitro arom nonortho fragments)
- `rdkit_pka__rdkit__fr_nitro_arom`: nitro arom fragment count (source family: rdkit; raw name: fr_nitro_arom; description: count of RDKit-recognized nitro arom fragments)
- `rdkit_pka__rdkit__fr_nitro`: nitro fragment count (source family: rdkit; raw name: fr_nitro; description: count of RDKit-recognized nitro fragments)
- `rdkit_pka__rdkit__fr_nitrile`: nitrile fragment count (source family: rdkit; raw name: fr_nitrile; description: count of RDKit-recognized nitrile fragments)
- `rdkit_pka__rdkit__fr_morpholine`: morpholine fragment count (source family: rdkit; raw name: fr_morpholine; description: count of RDKit-recognized morpholine fragments)
- `rdkit_pka__rdkit__fr_methoxy`: methoxy fragment count (source family: rdkit; raw name: fr_methoxy; description: count of RDKit-recognized methoxy fragments)
- `rdkit_pka__rdkit__fr_lactone`: lactone fragment count (source family: rdkit; raw name: fr_lactone; description: count of RDKit-recognized lactone fragments)
- `rdkit_pka__rdkit__fr_lactam`: lactam fragment count (source family: rdkit; raw name: fr_lactam; description: count of RDKit-recognized lactam fragments)
- `rdkit_pka__rdkit__fr_ketone_Topliss`: ketone Topliss fragment count (source family: rdkit; raw name: fr_ketone_Topliss; description: count of RDKit-recognized ketone Topliss fragments)
- `rdkit_pka__rdkit__fr_ketone`: ketone fragment count (source family: rdkit; raw name: fr_ketone; description: count of RDKit-recognized ketone fragments)
- `rdkit_pka__rdkit__fr_isothiocyan`: isothiocyan fragment count (source family: rdkit; raw name: fr_isothiocyan; description: count of RDKit-recognized isothiocyan fragments)
- `rdkit_pka__rdkit__fr_isocyan`: isocyan fragment count (source family: rdkit; raw name: fr_isocyan; description: count of RDKit-recognized isocyan fragments)
- `rdkit_pka__rdkit__fr_imide`: imide fragment count (source family: rdkit; raw name: fr_imide; description: count of RDKit-recognized imide fragments)
- `rdkit_pka__rdkit__fr_imidazole`: imidazole fragment count (source family: rdkit; raw name: fr_imidazole; description: count of RDKit-recognized imidazole fragments)
- `rdkit_pka__rdkit__fr_hdrzone`: hdrzone fragment count (source family: rdkit; raw name: fr_hdrzone; description: count of RDKit-recognized hdrzone fragments)
- `rdkit_pka__rdkit__fr_hdrzine`: hdrzine fragment count (source family: rdkit; raw name: fr_hdrzine; description: count of RDKit-recognized hdrzine fragments)
- `rdkit_pka__rdkit__fr_halogen`: halogen fragment count (source family: rdkit; raw name: fr_halogen; description: count of RDKit-recognized halogen fragments)
- `rdkit_pka__rdkit__fr_guanido`: guanido fragment count (source family: rdkit; raw name: fr_guanido; description: count of RDKit-recognized guanido fragments)
- `rdkit_pka__rdkit__fr_furan`: furan fragment count (source family: rdkit; raw name: fr_furan; description: count of RDKit-recognized furan fragments)
- `rdkit_pka__rdkit__fr_ether`: ether fragment count (source family: rdkit; raw name: fr_ether; description: count of RDKit-recognized ether fragments)
- `rdkit_pka__rdkit__fr_ester`: ester fragment count (source family: rdkit; raw name: fr_ester; description: count of RDKit-recognized ester fragments)
- `rdkit_pka__rdkit__fr_epoxide`: epoxide fragment count (source family: rdkit; raw name: fr_epoxide; description: count of RDKit-recognized epoxide fragments)
- `rdkit_pka__rdkit__fr_dihydropyridine`: dihydropyridine fragment count (source family: rdkit; raw name: fr_dihydropyridine; description: count of RDKit-recognized dihydropyridine fragments)
- `rdkit_pka__rdkit__fr_diazo`: diazo fragment count (source family: rdkit; raw name: fr_diazo; description: count of RDKit-recognized diazo fragments)
- `rdkit_pka__rdkit__fr_bicyclic`: bicyclic fragment count (source family: rdkit; raw name: fr_bicyclic; description: count of RDKit-recognized bicyclic fragments)
- `rdkit_pka__rdkit__fr_benzodiazepine`: benzodiazepine fragment count (source family: rdkit; raw name: fr_benzodiazepine; description: count of RDKit-recognized benzodiazepine fragments)
- `rdkit_pka__rdkit__fr_benzene`: benzene fragment count (source family: rdkit; raw name: fr_benzene; description: count of RDKit-recognized benzene fragments)
- `rdkit_pka__rdkit__fr_barbitur`: barbitur fragment count (source family: rdkit; raw name: fr_barbitur; description: count of RDKit-recognized barbitur fragments)
- `rdkit_pka__rdkit__fr_azo`: azo fragment count (source family: rdkit; raw name: fr_azo; description: count of RDKit-recognized azo fragments)
- `rdkit_pka__rdkit__fr_azide`: azide fragment count (source family: rdkit; raw name: fr_azide; description: count of RDKit-recognized azide fragments)
- `rdkit_pka__rdkit__fr_aryl_methyl`: aryl methyl fragment count (source family: rdkit; raw name: fr_aryl_methyl; description: count of RDKit-recognized aryl methyl fragments)
- `rdkit_pka__rdkit__fr_aniline`: aniline fragment count (source family: rdkit; raw name: fr_aniline; description: count of RDKit-recognized aniline fragments)
- `rdkit_pka__rdkit__fr_amidine`: amidine fragment count (source family: rdkit; raw name: fr_amidine; description: count of RDKit-recognized amidine fragments)
- `rdkit_pka__rdkit__fr_amide`: amide fragment count (source family: rdkit; raw name: fr_amide; description: count of RDKit-recognized amide fragments)
- `rdkit_pka__rdkit__fr_allylic_oxid`: allylic oxid fragment count (source family: rdkit; raw name: fr_allylic_oxid; description: count of RDKit-recognized allylic oxid fragments)
- `rdkit_pka__rdkit__fr_alkyl_halide`: alkyl halide fragment count (source family: rdkit; raw name: fr_alkyl_halide; description: count of RDKit-recognized alkyl halide fragments)
- `rdkit_pka__rdkit__fr_alkyl_carbamate`: alkyl carbamate fragment count (source family: rdkit; raw name: fr_alkyl_carbamate; description: count of RDKit-recognized alkyl carbamate fragments)
- `rdkit_pka__rdkit__fr_aldehyde`: aldehyde fragment count (source family: rdkit; raw name: fr_aldehyde; description: count of RDKit-recognized aldehyde fragments)
- `rdkit_pka__rdkit__fr_SH`: SH fragment count (source family: rdkit; raw name: fr_SH; description: count of RDKit-recognized SH fragments)
- `rdkit_pka__rdkit__fr_Nhpyrrole`: pyrrole-like NH nitrogen count (source family: rdkit; raw name: fr_Nhpyrrole; description: count of pyrrole-like nitrogens that carry a hydrogen)
- `rdkit_pka__rdkit__fr_Ndealkylation2`: Ndealkylation2 fragment count (source family: rdkit; raw name: fr_Ndealkylation2; description: count of RDKit-recognized Ndealkylation2 fragments)
- `rdkit_pka__rdkit__fr_Ndealkylation1`: Ndealkylation1 fragment count (source family: rdkit; raw name: fr_Ndealkylation1; description: count of RDKit-recognized Ndealkylation1 fragments)
- `rdkit_pka__rdkit__fr_N_O`: N-oxide fragment count (source family: rdkit; raw name: fr_N_O; description: count of N-oxide fragments)
- `rdkit_pka__rdkit__fr_NH2`: primary amine-like NH2 fragment count (source family: rdkit; raw name: fr_NH2; description: count of nitrogen fragments with two attached hydrogens)
- `rdkit_pka__rdkit__fr_NH1`: secondary amine-like NH1 fragment count (source family: rdkit; raw name: fr_NH1; description: count of nitrogen fragments with one attached hydrogen)
- `rdkit_pka__rdkit__fr_NH0`: tertiary amine-like NH0 fragment count (source family: rdkit; raw name: fr_NH0; description: count of tertiary amine-like nitrogen fragments with no hydrogens)
- `rdkit_pka__rdkit__fr_Imine`: imine fragment count (source family: rdkit; raw name: fr_Imine; description: count of imine-like C=N fragments)
- `rdkit_pka__rdkit__fr_HOCCN`: specific amino-alcohol motif count (source family: rdkit; raw name: fr_HOCCN; description: count of a specific HO-CC-N amino-alcohol-like substructure recognized by RDKit)
- `rdkit_pka__rdkit__fr_C_S`: C S fragment count (source family: rdkit; raw name: fr_C_S; description: count of RDKit-recognized C S fragments)
- `rdkit_pka__rdkit__fr_C_O_noCOO`: non-carboxyl C-O motif count (source family: rdkit; raw name: fr_C_O_noCOO; description: count of carbon-oxygen motifs excluding carboxylic-acid related ones)
- `rdkit_pka__rdkit__fr_C_O`: alcohol/ether C-O motif count (source family: rdkit; raw name: fr_C_O; description: count of alcohol or ether carbon-oxygen motifs)
- `rdkit_pka__rdkit__fr_COO2`: carboxylate fragment count (source family: rdkit; raw name: fr_COO2; description: count of carboxylate or deprotonated carboxylic acid fragments)
- `rdkit_pka__rdkit__fr_COO`: carboxylic acid fragment count (source family: rdkit; raw name: fr_COO; description: count of carboxylic acid related fragments)
- `rdkit_pka__rdkit__fr_Ar_OH`: Ar OH fragment count (source family: rdkit; raw name: fr_Ar_OH; description: count of RDKit-recognized Ar OH fragments)
- `rdkit_pka__rdkit__fr_Ar_NH`: Ar NH fragment count (source family: rdkit; raw name: fr_Ar_NH; description: count of RDKit-recognized Ar NH fragments)
- `rdkit_pka__rdkit__fr_Ar_N`: Ar N fragment count (source family: rdkit; raw name: fr_Ar_N; description: count of RDKit-recognized Ar N fragments)
- `rdkit_pka__rdkit__fr_Ar_COO`: aromatic carboxylic acid count (source family: rdkit; raw name: fr_Ar_COO; description: count of aromatic carboxylic acid fragments)
- `rdkit_pka__rdkit__fr_ArN`: ArN fragment count (source family: rdkit; raw name: fr_ArN; description: count of RDKit-recognized ArN fragments)
- `rdkit_pka__rdkit__fr_Al_OH_noTert`: non-tertiary aliphatic alcohol group count (source family: rdkit; raw name: fr_Al_OH_noTert; description: count of aliphatic hydroxyl groups excluding tertiary alcohols)
- `rdkit_pka__rdkit__fr_Al_OH`: aliphatic alcohol group count (source family: rdkit; raw name: fr_Al_OH; description: count of aliphatic hydroxyl or alcohol groups)
- `rdkit_pka__rdkit__fr_Al_COO`: aliphatic carboxylic acid count (source family: rdkit; raw name: fr_Al_COO; description: count of aliphatic carboxylic acid fragments)
- `rdkit_pka__rdkit__MinPartialCharge`: minimum partial charge (source family: rdkit; raw name: MinPartialCharge; description: most negative atomic partial charge)
- `rdkit_pka__rdkit__MinAbsPartialCharge`: minimum absolute partial charge (source family: rdkit; raw name: MinAbsPartialCharge; description: smallest absolute atomic partial charge)
- `rdkit_pka__rdkit__MaxPartialCharge`: maximum partial charge (source family: rdkit; raw name: MaxPartialCharge; description: most positive atomic partial charge)
- `rdkit_pka__rdkit__MaxAbsPartialCharge`: maximum absolute partial charge (source family: rdkit; raw name: MaxAbsPartialCharge; description: largest absolute atomic partial charge)
- `rdkit_pka__pka__warning_multiple_basic_sites`: multiple basic-site warning (source family: pka; raw name: warning_multiple_basic_sites; description: whether pKa estimation detected multiple basic sites)
- `rdkit_pka__pka__warning_multiple_acidic_sites`: multiple acidic-site warning (source family: pka; raw name: warning_multiple_acidic_sites; description: whether pKa estimation detected multiple acidic sites)
- `rdkit_pka__pka__warning_fraction_neutral_tiny`: very small neutral-fraction warning (source family: pka; raw name: warning_fraction_neutral_tiny; description: whether the estimated neutral fraction was extremely small)
- `rdkit_pka__pka__warning_fraction_neutral_clamped`: neutral-fraction clamped warning (source family: pka; raw name: warning_fraction_neutral_clamped; description: whether the estimated neutral fraction had to be clamped)
- `rdkit_pka__pka__warning_count`: pKa warning count (source family: pka; raw name: warning_count; description: number of warnings raised during pKa/logD estimation)
- `rdkit_pka__pka__warning_amphoteric`: amphoteric warning (source family: pka; raw name: warning_amphoteric; description: whether pKa estimation flagged the molecule as amphoteric)
- `rdkit_pka__pka__is_amphoteric`: is amphoteric (source family: pka; raw name: is_amphoteric; description: whether the molecule has both acidic and basic ionizable sites)
- `rdkit_pka__pka__has_basic_site`: has a basic site (source family: pka; raw name: has_basic_site; description: whether the molecule has at least one basic ionizable site)
- `rdkit_pka__pka__has_acidic_site`: has an acidic site (source family: pka; raw name: has_acidic_site; description: whether the molecule has at least one acidic ionizable site)

## Observed Threshold Examples From Reasoning Data

### `rdkit_pka__rdkit__qed`

- Display name: QED drug-likeness
- Description: quantitative estimate of drug-likeness
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_urea`

- Display name: urea fragment count
- Description: count of RDKit-recognized urea fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_unbrch_alkane`

- Display name: unbrch alkane fragment count
- Description: count of RDKit-recognized unbrch alkane fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_thiophene`

- Display name: thiophene fragment count
- Description: count of RDKit-recognized thiophene fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_thiocyan`

- Display name: thiocyan fragment count
- Description: count of RDKit-recognized thiocyan fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_thiazole`

- Display name: thiazole fragment count
- Description: count of RDKit-recognized thiazole fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_tetrazole`

- Display name: tetrazole fragment count
- Description: count of RDKit-recognized tetrazole fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_term_acetylene`

- Display name: term acetylene fragment count
- Description: count of RDKit-recognized term acetylene fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_sulfone`

- Display name: sulfone fragment count
- Description: count of RDKit-recognized sulfone fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_sulfonamd`

- Display name: sulfonamd fragment count
- Description: count of RDKit-recognized sulfonamd fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_sulfide`

- Display name: sulfide fragment count
- Description: count of RDKit-recognized sulfide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_quatN`

- Display name: quaternary nitrogen count
- Description: count of quaternary ammonium or quaternary nitrogen centers
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_pyridine`

- Display name: pyridine fragment count
- Description: count of RDKit-recognized pyridine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_prisulfonamd`

- Display name: prisulfonamd fragment count
- Description: count of RDKit-recognized prisulfonamd fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_priamide`

- Display name: priamide fragment count
- Description: count of RDKit-recognized priamide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_piperzine`

- Display name: piperzine fragment count
- Description: count of RDKit-recognized piperzine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_piperdine`

- Display name: piperdine fragment count
- Description: count of RDKit-recognized piperdine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_phos_ester`

- Display name: phos ester fragment count
- Description: count of RDKit-recognized phos ester fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_phos_acid`

- Display name: phos acid fragment count
- Description: count of RDKit-recognized phos acid fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_phenol_noOrthoHbond`

- Display name: phenol noOrthoHbond fragment count
- Description: count of RDKit-recognized phenol noOrthoHbond fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_phenol`

- Display name: phenol fragment count
- Description: count of RDKit-recognized phenol fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_para_hydroxylation`

- Display name: para hydroxylation fragment count
- Description: count of RDKit-recognized para hydroxylation fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_oxime`

- Display name: oxime fragment count
- Description: count of RDKit-recognized oxime fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_oxazole`

- Display name: oxazole fragment count
- Description: count of RDKit-recognized oxazole fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_nitroso`

- Display name: nitroso fragment count
- Description: count of RDKit-recognized nitroso fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_nitro_arom_nonortho`

- Display name: nitro arom nonortho fragment count
- Description: count of RDKit-recognized nitro arom nonortho fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_nitro_arom`

- Display name: nitro arom fragment count
- Description: count of RDKit-recognized nitro arom fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_nitro`

- Display name: nitro fragment count
- Description: count of RDKit-recognized nitro fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_nitrile`

- Display name: nitrile fragment count
- Description: count of RDKit-recognized nitrile fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_morpholine`

- Display name: morpholine fragment count
- Description: count of RDKit-recognized morpholine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_methoxy`

- Display name: methoxy fragment count
- Description: count of RDKit-recognized methoxy fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_lactone`

- Display name: lactone fragment count
- Description: count of RDKit-recognized lactone fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_lactam`

- Display name: lactam fragment count
- Description: count of RDKit-recognized lactam fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_ketone_Topliss`

- Display name: ketone Topliss fragment count
- Description: count of RDKit-recognized ketone Topliss fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_ketone`

- Display name: ketone fragment count
- Description: count of RDKit-recognized ketone fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_isothiocyan`

- Display name: isothiocyan fragment count
- Description: count of RDKit-recognized isothiocyan fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_isocyan`

- Display name: isocyan fragment count
- Description: count of RDKit-recognized isocyan fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_imide`

- Display name: imide fragment count
- Description: count of RDKit-recognized imide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_imidazole`

- Display name: imidazole fragment count
- Description: count of RDKit-recognized imidazole fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_hdrzone`

- Display name: hdrzone fragment count
- Description: count of RDKit-recognized hdrzone fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_hdrzine`

- Display name: hdrzine fragment count
- Description: count of RDKit-recognized hdrzine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_halogen`

- Display name: halogen fragment count
- Description: count of RDKit-recognized halogen fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_guanido`

- Display name: guanido fragment count
- Description: count of RDKit-recognized guanido fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_furan`

- Display name: furan fragment count
- Description: count of RDKit-recognized furan fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_ether`

- Display name: ether fragment count
- Description: count of RDKit-recognized ether fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_ester`

- Display name: ester fragment count
- Description: count of RDKit-recognized ester fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_epoxide`

- Display name: epoxide fragment count
- Description: count of RDKit-recognized epoxide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_dihydropyridine`

- Display name: dihydropyridine fragment count
- Description: count of RDKit-recognized dihydropyridine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_diazo`

- Display name: diazo fragment count
- Description: count of RDKit-recognized diazo fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_bicyclic`

- Display name: bicyclic fragment count
- Description: count of RDKit-recognized bicyclic fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_benzodiazepine`

- Display name: benzodiazepine fragment count
- Description: count of RDKit-recognized benzodiazepine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_benzene`

- Display name: benzene fragment count
- Description: count of RDKit-recognized benzene fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_barbitur`

- Display name: barbitur fragment count
- Description: count of RDKit-recognized barbitur fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_azo`

- Display name: azo fragment count
- Description: count of RDKit-recognized azo fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_azide`

- Display name: azide fragment count
- Description: count of RDKit-recognized azide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_aryl_methyl`

- Display name: aryl methyl fragment count
- Description: count of RDKit-recognized aryl methyl fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_aniline`

- Display name: aniline fragment count
- Description: count of RDKit-recognized aniline fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_amidine`

- Display name: amidine fragment count
- Description: count of RDKit-recognized amidine fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_amide`

- Display name: amide fragment count
- Description: count of RDKit-recognized amide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_allylic_oxid`

- Display name: allylic oxid fragment count
- Description: count of RDKit-recognized allylic oxid fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_alkyl_halide`

- Display name: alkyl halide fragment count
- Description: count of RDKit-recognized alkyl halide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_alkyl_carbamate`

- Display name: alkyl carbamate fragment count
- Description: count of RDKit-recognized alkyl carbamate fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_aldehyde`

- Display name: aldehyde fragment count
- Description: count of RDKit-recognized aldehyde fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_SH`

- Display name: SH fragment count
- Description: count of RDKit-recognized SH fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Nhpyrrole`

- Display name: pyrrole-like NH nitrogen count
- Description: count of pyrrole-like nitrogens that carry a hydrogen
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ndealkylation2`

- Display name: Ndealkylation2 fragment count
- Description: count of RDKit-recognized Ndealkylation2 fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ndealkylation1`

- Display name: Ndealkylation1 fragment count
- Description: count of RDKit-recognized Ndealkylation1 fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_N_O`

- Display name: N-oxide fragment count
- Description: count of N-oxide fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_NH2`

- Display name: primary amine-like NH2 fragment count
- Description: count of nitrogen fragments with two attached hydrogens
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_NH1`

- Display name: secondary amine-like NH1 fragment count
- Description: count of nitrogen fragments with one attached hydrogen
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_NH0`

- Display name: tertiary amine-like NH0 fragment count
- Description: count of tertiary amine-like nitrogen fragments with no hydrogens
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Imine`

- Display name: imine fragment count
- Description: count of imine-like C=N fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_HOCCN`

- Display name: specific amino-alcohol motif count
- Description: count of a specific HO-CC-N amino-alcohol-like substructure recognized by RDKit
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_C_S`

- Display name: C S fragment count
- Description: count of RDKit-recognized C S fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_C_O_noCOO`

- Display name: non-carboxyl C-O motif count
- Description: count of carbon-oxygen motifs excluding carboxylic-acid related ones
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_C_O`

- Display name: alcohol/ether C-O motif count
- Description: count of alcohol or ether carbon-oxygen motifs
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_COO2`

- Display name: carboxylate fragment count
- Description: count of carboxylate or deprotonated carboxylic acid fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_COO`

- Display name: carboxylic acid fragment count
- Description: count of carboxylic acid related fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ar_OH`

- Display name: Ar OH fragment count
- Description: count of RDKit-recognized Ar OH fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ar_NH`

- Display name: Ar NH fragment count
- Description: count of RDKit-recognized Ar NH fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ar_N`

- Display name: Ar N fragment count
- Description: count of RDKit-recognized Ar N fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Ar_COO`

- Display name: aromatic carboxylic acid count
- Description: count of aromatic carboxylic acid fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_ArN`

- Display name: ArN fragment count
- Description: count of RDKit-recognized ArN fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Al_OH_noTert`

- Display name: non-tertiary aliphatic alcohol group count
- Description: count of aliphatic hydroxyl groups excluding tertiary alcohols
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Al_OH`

- Display name: aliphatic alcohol group count
- Description: count of aliphatic hydroxyl or alcohol groups
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__fr_Al_COO`

- Display name: aliphatic carboxylic acid count
- Description: count of aliphatic carboxylic acid fragments
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: RDKit fragment-count features are usually better used as qualitative structural evidence than as literature-threshold targets
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__TPSA`

- Display name: topological polar surface area
- Description: topological polar surface area of the molecule
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__RingCount`

- Display name: ring count
- Description: total number of rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumSaturatedRings`

- Display name: saturated ring count
- Description: number of saturated rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumSaturatedHeterocycles`

- Display name: saturated heterocycle count
- Description: number of saturated heterocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumSaturatedCarbocycles`

- Display name: saturated carbocycle count
- Description: number of saturated carbocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumRotatableBonds`

- Display name: rotatable-bond count
- Description: number of rotatable bonds
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumHeteroatoms`

- Display name: heteroatom count
- Description: number of heteroatoms, such as N, O, or S
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumHDonors`

- Display name: hydrogen-bond donor count
- Description: number of hydrogen-bond donors
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumHAcceptors`

- Display name: hydrogen-bond acceptor count
- Description: number of hydrogen-bond acceptors
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAromaticRings`

- Display name: aromatic ring count
- Description: number of aromatic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAromaticHeterocycles`

- Display name: aromatic heterocycle count
- Description: number of aromatic heterocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAromaticCarbocycles`

- Display name: aromatic carbocycle count
- Description: number of aromatic carbocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAliphaticRings`

- Display name: aliphatic ring count
- Description: number of aliphatic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAliphaticHeterocycles`

- Display name: aliphatic heterocycle count
- Description: number of aliphatic heterocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NumAliphaticCarbocycles`

- Display name: aliphatic carbocycle count
- Description: number of aliphatic carbocyclic rings
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NOCount`

- Display name: nitrogen/oxygen atom count
- Description: number of nitrogen and oxygen atoms
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__NHOHCount`

- Display name: NH/OH group count
- Description: number of NH or OH groups
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MolWt`

- Display name: molecular weight
- Description: molecular weight
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MolLogP`

- Display name: estimated logP
- Description: RDKit-estimated octanol/water partition coefficient (logP)
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MinPartialCharge`

- Display name: minimum partial charge
- Description: most negative atomic partial charge
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MinAbsPartialCharge`

- Display name: minimum absolute partial charge
- Description: smallest absolute atomic partial charge
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MaxPartialCharge`

- Display name: maximum partial charge
- Description: most positive atomic partial charge
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__MaxAbsPartialCharge`

- Display name: maximum absolute partial charge
- Description: largest absolute atomic partial charge
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: descriptor is useful for reasoning but less likely to have a stable literature threshold for direct substitution
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__LabuteASA`

- Display name: Labute surface area
- Description: Labute approximate surface area
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__HeavyAtomMolWt`

- Display name: heavy-atom molecular weight
- Description: molecular weight contributed by heavy atoms
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__HeavyAtomCount`

- Display name: heavy-atom count
- Description: number of non-hydrogen atoms
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__FractionCSP3`

- Display name: fraction of sp3 carbons
- Description: fraction of carbon atoms that are sp3 hybridized
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__rdkit__ExactMolWt`

- Display name: exact molecular weight
- Description: exact isotopic molecular weight
- Source family: rdkit
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: classic RDKit physicochemical or count descriptor that often appears in literature threshold heuristics
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_multiple_basic_sites`

- Display name: multiple basic-site warning
- Description: whether pKa estimation detected multiple basic sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_multiple_acidic_sites`

- Display name: multiple acidic-site warning
- Description: whether pKa estimation detected multiple acidic sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_fraction_neutral_tiny`

- Display name: very small neutral-fraction warning
- Description: whether the estimated neutral fraction was extremely small
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_fraction_neutral_clamped`

- Display name: neutral-fraction clamped warning
- Description: whether the estimated neutral fraction had to be clamped
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_count`

- Display name: pKa warning count
- Description: number of warnings raised during pKa/logD estimation
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__warning_amphoteric`

- Display name: amphoteric warning
- Description: whether pKa estimation flagged the molecule as amphoteric
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__num_ionizable_sites`

- Display name: number of ionizable sites
- Description: total number of acidic and basic ionizable sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__num_basic_sites`

- Display name: number of basic sites
- Description: number of basic ionizable sites in the molecule
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__num_acidic_sites`

- Display name: number of acidic sites
- Description: number of acidic ionizable sites in the molecule
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__logp_wildman_crippen`

- Display name: estimated logP
- Description: Wildman-Crippen estimated logP
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__logd_ph`

- Display name: logD pH setting
- Description: pH value used when estimating logD
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__logd_estimate`

- Display name: estimated logD
- Description: estimated logD at the configured pH
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__is_amphoteric`

- Display name: is amphoteric
- Description: whether the molecule has both acidic and basic ionizable sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__has_basic_site`

- Display name: has a basic site
- Description: whether the molecule has at least one basic ionizable site
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__has_acidic_site`

- Display name: has an acidic site
- Description: whether the molecule has at least one acidic ionizable site
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: qualitative_rewrite
- Track rationale: pKa warning or boolean state feature is better used as qualitative support than as a threshold target
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__fraction_neutral`

- Display name: neutral fraction
- Description: estimated fraction of the molecule that is neutral at the configured pH
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__base_site_pka_sum`

- Display name: sum basic site pKa
- Description: sum pKa across the basic ionizable sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.

### `rdkit_pka__pka__acid_site_pka_sum`

- Display name: sum acidic site pKa
- Description: sum pKa across the acidic ionizable sites
- Source family: pka
- In surviving feature universe: True
- Observed in reasoning JSON: False
- Research track: threshold_research
- Track rationale: pKa/logD/logP/site-count style feature is often suitable for literature threshold or range research
- Important-feature occurrences: 0
- Reasoning-step occurrences: 0
- Sample count: 0

- Threshold examples: none captured in selected reasoning steps.
