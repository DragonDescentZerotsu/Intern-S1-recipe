from __future__ import annotations

import re


PKA_EXACT_DESCRIPTIONS = {
    "num_basic_sites": ("number of basic sites", "number of basic ionizable sites in the molecule"),
    "num_acidic_sites": ("number of acidic sites", "number of acidic ionizable sites in the molecule"),
    "num_ionizable_sites": ("number of ionizable sites", "total number of acidic and basic ionizable sites"),
    "has_basic_site": ("has a basic site", "whether the molecule has at least one basic ionizable site"),
    "has_acidic_site": ("has an acidic site", "whether the molecule has at least one acidic ionizable site"),
    "is_amphoteric": ("is amphoteric", "whether the molecule has both acidic and basic ionizable sites"),
    "most_basic_pka": ("strongest basic pKa", "pKa of the strongest basic site"),
    "most_acidic_pka": ("strongest acidic pKa", "pKa of the strongest acidic site"),
    "logd_ph": ("logD pH setting", "pH value used when estimating logD"),
    "logp_wildman_crippen": ("estimated logP", "Wildman-Crippen estimated logP"),
    "logd_estimate": ("estimated logD", "estimated logD at the configured pH"),
    "fraction_neutral": ("neutral fraction", "estimated fraction of the molecule that is neutral at the configured pH"),
    "warning_count": ("pKa warning count", "number of warnings raised during pKa/logD estimation"),
    "warning_multiple_basic_sites": (
        "multiple basic-site warning",
        "whether pKa estimation detected multiple basic sites",
    ),
    "warning_multiple_acidic_sites": (
        "multiple acidic-site warning",
        "whether pKa estimation detected multiple acidic sites",
    ),
    "warning_amphoteric": (
        "amphoteric warning",
        "whether pKa estimation flagged the molecule as amphoteric",
    ),
    "warning_fraction_neutral_clamped": (
        "neutral-fraction clamped warning",
        "whether the estimated neutral fraction had to be clamped",
    ),
    "warning_fraction_neutral_tiny": (
        "very small neutral-fraction warning",
        "whether the estimated neutral fraction was extremely small",
    ),
}


RDKIT_EXACT_DESCRIPTIONS = {
    "TPSA": ("topological polar surface area", "topological polar surface area of the molecule"),
    "MolWt": ("molecular weight", "molecular weight"),
    "HeavyAtomMolWt": ("heavy-atom molecular weight", "molecular weight contributed by heavy atoms"),
    "ExactMolWt": ("exact molecular weight", "exact isotopic molecular weight"),
    "MolLogP": ("estimated logP", "RDKit-estimated octanol/water partition coefficient (logP)"),
    "qed": ("QED drug-likeness", "quantitative estimate of drug-likeness"),
    "NumHDonors": ("hydrogen-bond donor count", "number of hydrogen-bond donors"),
    "NumHAcceptors": ("hydrogen-bond acceptor count", "number of hydrogen-bond acceptors"),
    "NumHeteroatoms": ("heteroatom count", "number of heteroatoms, such as N, O, or S"),
    "NumRotatableBonds": ("rotatable-bond count", "number of rotatable bonds"),
    "HeavyAtomCount": ("heavy-atom count", "number of non-hydrogen atoms"),
    "NHOHCount": ("NH/OH group count", "number of NH or OH groups"),
    "NOCount": ("nitrogen/oxygen atom count", "number of nitrogen and oxygen atoms"),
    "RingCount": ("ring count", "total number of rings"),
    "NumAromaticRings": ("aromatic ring count", "number of aromatic rings"),
    "NumAliphaticRings": ("aliphatic ring count", "number of aliphatic rings"),
    "NumSaturatedRings": ("saturated ring count", "number of saturated rings"),
    "NumAromaticCarbocycles": ("aromatic carbocycle count", "number of aromatic carbocyclic rings"),
    "NumAromaticHeterocycles": ("aromatic heterocycle count", "number of aromatic heterocyclic rings"),
    "NumAliphaticCarbocycles": ("aliphatic carbocycle count", "number of aliphatic carbocyclic rings"),
    "NumAliphaticHeterocycles": ("aliphatic heterocycle count", "number of aliphatic heterocyclic rings"),
    "NumSaturatedCarbocycles": ("saturated carbocycle count", "number of saturated carbocyclic rings"),
    "NumSaturatedHeterocycles": ("saturated heterocycle count", "number of saturated heterocyclic rings"),
    "FractionCSP3": ("fraction of sp3 carbons", "fraction of carbon atoms that are sp3 hybridized"),
    "LabuteASA": ("Labute surface area", "Labute approximate surface area"),
    "BertzCT": ("Bertz complexity", "Bertz topological complexity index"),
    "HallKierAlpha": ("Hall-Kier alpha", "Hall-Kier alpha shape parameter"),
    "MinPartialCharge": ("minimum partial charge", "most negative atomic partial charge"),
    "MaxPartialCharge": ("maximum partial charge", "most positive atomic partial charge"),
    "MinAbsPartialCharge": ("minimum absolute partial charge", "smallest absolute atomic partial charge"),
    "MaxAbsPartialCharge": ("maximum absolute partial charge", "largest absolute atomic partial charge"),
    "MinEStateIndex": ("minimum EState index", "minimum electrotopological state index"),
    "MaxEStateIndex": ("maximum EState index", "maximum electrotopological state index"),
    "MinAbsEStateIndex": ("minimum absolute EState index", "smallest absolute electrotopological state index"),
    "MaxAbsEStateIndex": ("maximum absolute EState index", "largest absolute electrotopological state index"),
}


BCUT_SUFFIX_DESCRIPTIONS = {
    "CHGLO": ("BCUT low charge eigenvalue", "BCUT2D low eigenvalue based on partial charge"),
    "CHGHI": ("BCUT high charge eigenvalue", "BCUT2D high eigenvalue based on partial charge"),
    "LOGPLOW": ("BCUT low logP eigenvalue", "BCUT2D low eigenvalue based on logP contribution"),
    "LOGPHI": ("BCUT high logP eigenvalue", "BCUT2D high eigenvalue based on logP contribution"),
    "MRLOW": ("BCUT low molar-refractivity eigenvalue", "BCUT2D low eigenvalue based on molar refractivity"),
    "MRHI": ("BCUT high molar-refractivity eigenvalue", "BCUT2D high eigenvalue based on molar refractivity"),
    "MWLOW": ("BCUT low mass eigenvalue", "BCUT2D low eigenvalue based on atomic mass"),
    "MWHI": ("BCUT high mass eigenvalue", "BCUT2D high eigenvalue based on atomic mass"),
}


SURFACE_AREA_PREFIX_DESCRIPTIONS = {
    "PEOE_VSA": ("partial-charge surface area bin", "van der Waals surface area in a partial-charge bin"),
    "SMR_VSA": ("molar-refractivity surface area bin", "van der Waals surface area in a molar-refractivity bin"),
    "SlogP_VSA": ("logP surface area bin", "van der Waals surface area in a logP bin"),
    "EState_VSA": ("EState-weighted surface area bin", "electrotopological-state-weighted surface area bin"),
    "VSA_EState": ("surface-area-weighted EState bin", "surface-area-weighted electrotopological-state bin"),
}


FRAGMENT_EXACT_DESCRIPTIONS = {
    "fr_COO": ("carboxylic acid fragment count", "count of carboxylic acid related fragments"),
    "fr_COO2": ("carboxylate fragment count", "count of carboxylate or deprotonated carboxylic acid fragments"),
    "fr_Al_COO": ("aliphatic carboxylic acid count", "count of aliphatic carboxylic acid fragments"),
    "fr_Al_OH": ("aliphatic alcohol group count", "count of aliphatic hydroxyl or alcohol groups"),
    "fr_Al_OH_noTert": (
        "non-tertiary aliphatic alcohol group count",
        "count of aliphatic hydroxyl groups excluding tertiary alcohols",
    ),
    "fr_Ar_COO": ("aromatic carboxylic acid count", "count of aromatic carboxylic acid fragments"),
    "fr_C_O": ("alcohol/ether C-O motif count", "count of alcohol or ether carbon-oxygen motifs"),
    "fr_C_O_noCOO": (
        "non-carboxyl C-O motif count",
        "count of carbon-oxygen motifs excluding carboxylic-acid related ones",
    ),
    "fr_HOCCN": (
        "specific amino-alcohol motif count",
        "count of a specific HO-CC-N amino-alcohol-like substructure recognized by RDKit",
    ),
    "fr_Imine": ("imine fragment count", "count of imine-like C=N fragments"),
    "fr_Nhpyrrole": ("pyrrole-like NH nitrogen count", "count of pyrrole-like nitrogens that carry a hydrogen"),
    "fr_NH0": ("tertiary amine-like NH0 fragment count", "count of tertiary amine-like nitrogen fragments with no hydrogens"),
    "fr_NH1": ("secondary amine-like NH1 fragment count", "count of nitrogen fragments with one attached hydrogen"),
    "fr_NH2": ("primary amine-like NH2 fragment count", "count of nitrogen fragments with two attached hydrogens"),
    "fr_N_O": ("N-oxide fragment count", "count of N-oxide fragments"),
    "fr_quatN": ("quaternary nitrogen count", "count of quaternary ammonium or quaternary nitrogen centers"),
}


TOKEN_REPLACEMENTS = {
    "logd": "logD",
    "logp": "logP",
    "pka": "pKa",
    "smr": "molar refractivity",
    "vsa": "surface area",
    "estate": "EState",
    "mw": "molecular weight",
    "chg": "charge",
    "csp3": "sp3 carbon",
    "tpsa": "topological polar surface area",
    "qed": "QED",
    "h": "hydrogen",
}

NATURAL_LANGUAGE_READY_RDKIT = {
    "TPSA",
    "MolWt",
    "HeavyAtomMolWt",
    "ExactMolWt",
    "MolLogP",
    "qed",
    "NumHDonors",
    "NumHAcceptors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "FractionCSP3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
}

TOO_TECHNICAL_PREFIXES = (
    "BCUT2D_",
    "PEOE_VSA",
    "SMR_VSA",
    "SlogP_VSA",
    "EState_VSA",
    "VSA_EState",
    "Chi",
    "Kappa",
)

TOO_TECHNICAL_EXACT = {
    "BalabanJ",
    "BertzCT",
    "HallKierAlpha",
    "Ipc",
    "AvgIpc",
    "MinEStateIndex",
    "MaxEStateIndex",
    "MinAbsEStateIndex",
    "MaxAbsEStateIndex",
}


def _titleize_tokens(tokens: list[str]) -> str:
    rendered = []
    for token in tokens:
        lowered = token.lower()
        rendered.append(TOKEN_REPLACEMENTS.get(lowered, token))
    return " ".join(rendered)


def _describe_pka_feature(local_name: str) -> tuple[str, str]:
    if local_name in PKA_EXACT_DESCRIPTIONS:
        return PKA_EXACT_DESCRIPTIONS[local_name]

    rank_match = re.fullmatch(r"(base_site_pka_rank|acid_site_pka_rank)_(\d+)", local_name)
    if rank_match:
        family, rank = rank_match.groups()
        site_kind = "basic" if family.startswith("base") else "acidic"
        return (
            f"{site_kind} site pKa rank {rank}",
            f"pKa value of the rank-{rank} {site_kind} ionizable site",
        )

    stats_match = re.fullmatch(r"(base_site_pka|acid_site_pka)_(min|max|mean|std|sum|range)", local_name)
    if stats_match:
        family, stat_name = stats_match.groups()
        site_kind = "basic" if family.startswith("base") else "acidic"
        stat_text = {
            "min": "minimum",
            "max": "maximum",
            "mean": "mean",
            "std": "standard deviation",
            "sum": "sum",
            "range": "range",
        }[stat_name]
        return (
            f"{stat_text} {site_kind} site pKa",
            f"{stat_text} pKa across the {site_kind} ionizable sites",
        )

    return (_titleize_tokens(local_name.split("_")), f"pKa-derived feature: {local_name}")


def _describe_rdkit_feature(local_name: str) -> tuple[str, str]:
    if local_name in RDKIT_EXACT_DESCRIPTIONS:
        return RDKIT_EXACT_DESCRIPTIONS[local_name]
    if local_name in FRAGMENT_EXACT_DESCRIPTIONS:
        return FRAGMENT_EXACT_DESCRIPTIONS[local_name]

    if local_name.startswith("fr_"):
        fragment_name = local_name[len("fr_"):].replace("_", " ")
        return (
            f"{fragment_name} fragment count",
            f"count of RDKit-recognized {fragment_name} fragments",
        )

    bcut_match = re.fullmatch(r"BCUT2D_(.+)", local_name)
    if bcut_match:
        suffix = bcut_match.group(1)
        if suffix in BCUT_SUFFIX_DESCRIPTIONS:
            return BCUT_SUFFIX_DESCRIPTIONS[suffix]

    for prefix, (display_root, description_root) in SURFACE_AREA_PREFIX_DESCRIPTIONS.items():
        bin_match = re.fullmatch(rf"{prefix}(\d+)", local_name)
        if bin_match:
            bin_index = bin_match.group(1)
            return (
                f"{display_root} {bin_index}",
                f"{description_root} {bin_index}",
            )

    chi_match = re.fullmatch(r"Chi(\d+)([A-Za-z]*)", local_name)
    if chi_match:
        order, variant = chi_match.groups()
        variant_text = {
            "v": "valence",
            "n": "connectivity",
        }.get(variant, variant or "standard")
        return (
            f"Chi {order} {variant_text} index",
            f"Hall-Kier Chi connectivity index of order {order} ({variant_text} variant)",
        )

    kappa_match = re.fullmatch(r"Kappa(\d+)", local_name)
    if kappa_match:
        order = kappa_match.group(1)
        return (
            f"Kappa {order} shape index",
            f"Kappa molecular shape index of order {order}",
        )

    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", local_name)
    return (spaced, f"RDKit descriptor: {spaced}")


def describe_feature_name(feature_name: str) -> dict[str, str]:
    if feature_name.startswith("fg_top_level__"):
        local_name = feature_name.split("__", 1)[1]
        return {
            "feature_name": feature_name,
            "display_name": local_name,
            "description": f"top-level functional-group indicator or count for {local_name}",
            "source_family": "fg_top_level",
            "raw_name": local_name,
        }

    if feature_name.startswith("rdkit_pka__pka__"):
        local_name = feature_name.split("__", 2)[2]
        display_name, description = _describe_pka_feature(local_name)
        return {
            "feature_name": feature_name,
            "display_name": display_name,
            "description": description,
            "source_family": "pka",
            "raw_name": local_name,
        }

    if feature_name.startswith("rdkit_pka__rdkit__"):
        local_name = feature_name.split("__", 2)[2]
        display_name, description = _describe_rdkit_feature(local_name)
        return {
            "feature_name": feature_name,
            "display_name": display_name,
            "description": description,
            "source_family": "rdkit",
            "raw_name": local_name,
        }

    if "__" in feature_name:
        local_name = feature_name.split("__", 1)[1]
    else:
        local_name = feature_name
    return {
        "feature_name": feature_name,
        "display_name": local_name.replace("_", " "),
        "description": f"model feature: {local_name}",
        "source_family": "generic",
        "raw_name": local_name,
    }


def classify_feature_nlp_readiness(feature_name: str) -> str:
    if feature_name.startswith("rdkit_pka__pka__"):
        return "natural_language_ready"

    if feature_name.startswith("rdkit_pka__rdkit__"):
        local_name = feature_name.split("__", 2)[2]
        if local_name.startswith("fr_"):
            return "natural_language_ready"
        if local_name in NATURAL_LANGUAGE_READY_RDKIT:
            return "natural_language_ready"
        if local_name in TOO_TECHNICAL_EXACT:
            return "too_technical_for_direct_reasoning"
        if local_name.startswith(TOO_TECHNICAL_PREFIXES):
            return "too_technical_for_direct_reasoning"
        return "needs_translation"

    if feature_name.startswith("fg_top_level__"):
        return "natural_language_ready"

    return "needs_translation"


def build_feature_semantics_map(feature_names: list[str]) -> dict[str, dict[str, str]]:
    semantics_map = {}
    for feature_name in feature_names:
        semantics = describe_feature_name(feature_name)
        semantics["nlp_readiness"] = classify_feature_nlp_readiness(feature_name)
        semantics_map[feature_name] = semantics
    return semantics_map
