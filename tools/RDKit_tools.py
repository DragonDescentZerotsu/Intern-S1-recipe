from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, GraphDescriptors, Fragments
from typing import Dict, Any, List


# -------------------------
# Core helper
# -------------------------
def _mol_from_smiles(smiles: str) -> Chem.Mol:
    """
    Parse a SMILES string into an RDKit Mol (sanitized 2D graph).
    Raises ValueError if SMILES is invalid or cannot be sanitized.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol

# -------------------------
# OpenAI “tools” schema (ChatCompletions / Responses compatible)
# -------------------------
def _tool(name: str, description: str) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the molecule."
                    }
                },
                "required": ["smiles"],
                "additionalProperties": False
            }
        }
    }


# ============================================================
# A) BASIC: widely useful across almost all TDC ADME/Tox tasks
# ============================================================
def get_molecular_weight(smiles: str) -> str:
    """Average molecular weight (Daltons) using RDKit's MolWt."""
    return f"Average molecular weight (Daltons): {float(Descriptors.MolWt(_mol_from_smiles(smiles))):.4f}"

def get_exact_molecular_weight(smiles: str) -> str:
    """Monoisotopic (exact) molecular weight (Daltons) using RDKit's ExactMolWt."""
    return f"Exact molecular weight (Daltons): {float(Descriptors.ExactMolWt(_mol_from_smiles(smiles))):.4f}"

def get_heavy_atom_count(smiles: str) -> str:
    """Number of non-hydrogen atoms (heavy atoms)."""
    return f"Number of heavy atoms: {int(Descriptors.HeavyAtomCount(_mol_from_smiles(smiles)))}"

def get_mol_logp(smiles: str) -> str:
    """
    Wildman–Crippen cLogP estimate (octanol/water partition coefficient).
    Computed from atom fragments + corrections (2D; no 3D needed).
    """
    return f"cLogP estimate: {float(Crippen.MolLogP(_mol_from_smiles(smiles))):.4f}"

def get_tpsa(smiles: str) -> str:
    """
    Topological Polar Surface Area (TPSA, Å^2) using RDKit's fragment-based method.
    Purely 2D/topological; no 3D needed.
    """
    return f"Topological Polar Surface Area (TPSA, Å^2): {float(rdMolDescriptors.CalcTPSA(_mol_from_smiles(smiles))):.4f}"

def get_hbd(smiles: str) -> str:
    """H-bond donors count (Lipinski-style SMARTS rules)."""
    return f"Number of H-bond donors: {int(Lipinski.NumHDonors(_mol_from_smiles(smiles)))}"

def get_hba(smiles: str) -> str:
    """H-bond acceptors count (Lipinski-style SMARTS rules)."""
    return f"Number of H-bond acceptors: {int(Lipinski.NumHAcceptors(_mol_from_smiles(smiles)))}"

def get_num_rotatable_bonds(smiles: str) -> str:
    """
    Rotatable bonds count (Lipinski-style definition; excludes e.g. amide C–N, ring bonds, etc.).
    """
    return f"Number of rotatable bonds: {int(Lipinski.NumRotatableBonds(_mol_from_smiles(smiles)))}"

def get_fraction_csp3(smiles: str) -> str:
    """
    FractionCSP3: (# of sp3 carbon atoms) / (total carbon atoms).
    Often used as a simple “3D character / saturation” proxy from 2D structure.
    """
    return f"Fraction of sp3 carbons: {float(rdMolDescriptors.CalcFractionCSP3(_mol_from_smiles(smiles))):.4f}"

def get_mol_mr(smiles: str) -> str:
    """
    Wildman–Crippen molar refractivity (MR) estimate.
    Proxy for polarizability/volume; fragment-based (2D).
    """
    return f"Wildman–Crippen molar refractivity (MR) estimate: {float(Crippen.MolMR(_mol_from_smiles(smiles))):.4f}"

def get_ring_count(smiles: str) -> str:
    """Total ring count. RDKit: Lipinski.RingCount(mol)."""
    v = int(Lipinski.RingCount(_mol_from_smiles(smiles)))
    return f"Total ring count: {v}"

def get_num_aromatic_rings(smiles: str) -> str:
    """Aromatic ring count. RDKit: Lipinski.NumAromaticRings(mol)."""
    v = int(Lipinski.NumAromaticRings(_mol_from_smiles(smiles)))
    return f"Aromatic ring count: {v}"

def get_formal_charge(smiles: str) -> str:
    """
    Net formal charge (sum of atom formal charges as encoded in the SMILES).
    RDKit: Chem.GetFormalCharge(mol).
    """
    v = int(Chem.GetFormalCharge(_mol_from_smiles(smiles)))
    return f"Net formal charge: {v}"

def get_qed(smiles: str) -> str:
    """
    QED (Quantitative Estimate of Drug-likeness) as implemented in RDKit.
    RDKit: QED.qed(mol). Returns a score in [0, 1].
    """
    v = float(QED.qed(_mol_from_smiles(smiles)))
    return f"QED (drug-likeness score, 0-1): {v:.4f}"

def get_num_heteroatoms(smiles: str) -> str:
    """Total heteroatom count (non C/H). RDKit: Lipinski.NumHeteroatoms(mol)."""
    v = int(Lipinski.NumHeteroatoms(_mol_from_smiles(smiles)))
    return f"Heteroatom count (non C/H): {v}"

# ============================================================
# 1) BASIC tool list (always include)
# ============================================================

RDKIT_BASIC_OPENAI_TOOLS = [
    _tool("get_molecular_weight", "Return the average molecular weight (Daltons)."),
    _tool("get_exact_molecular_weight", "Return the monoisotopic exact molecular weight (Daltons)."),
    _tool("get_heavy_atom_count", "Return the number of non-hydrogen atoms (heavy atoms)."),
    _tool("get_mol_logp", "Return Wildman–Crippen cLogP (octanol/water)."),
    _tool("get_tpsa", "Return topological polar surface area TPSA (Å^2)."),
    _tool("get_hbd", "Return Lipinski H-bond donors count."),
    _tool("get_hba", "Return Lipinski H-bond acceptors count."),
    _tool("get_num_rotatable_bonds", "Return rotatable bonds count."),
    _tool("get_fraction_csp3", "Return FractionCSP3 = (# sp3 carbon atoms)/(# total carbon atoms), Often used as a simple “3D character / saturation” proxy from 2D structure."),
    _tool("get_mol_mr", "Return Wildman–Crippen molar refractivity (MR)."),
    _tool("get_ring_count", "Return total ring count."),
    _tool("get_num_aromatic_rings", "Return aromatic ring count."),
    _tool("get_formal_charge", "Return net formal charge (sum of atom formal charges as encoded in the SMILES)."),
    _tool("get_qed", "Return QED (Quantitative Estimate of Drug-likeness) as implemented in RDKit."),
    _tool("get_num_heteroatoms", "Return heteroatom count (non C/H)."),
]

# ============================================================
# B) “SPECIFIC” building blocks (no fragments, still SMILES-only)
#    These get mixed into task-specific packs below.
# ============================================================

# --- Permeability / surface proxy ---
def get_labute_asa(smiles: str) -> str:
    """
    Labute approximate surface area (ASA proxy; topology/fragment-based, not true 3D SASA).
    RDKit: rdMolDescriptors.CalcLabuteASA(mol).
    """
    v = float(rdMolDescriptors.CalcLabuteASA(_mol_from_smiles(smiles)))
    return f"Labute approximate surface area (ASA proxy): {v:.4f}"

# --- Charge distribution proxy (Gasteiger; SMILES-only) ---
def get_max_abs_partial_charge(smiles: str) -> str:
    """Maximum absolute Gasteiger partial charge among atoms. RDKit: Descriptors.MaxAbsPartialCharge(mol)."""
    v = float(Descriptors.MaxAbsPartialCharge(_mol_from_smiles(smiles)))
    return f"Max absolute Gasteiger partial charge: {v:.4f}"

def get_min_abs_partial_charge(smiles: str) -> str:
    """Minimum absolute Gasteiger partial charge among atoms. RDKit: Descriptors.MinAbsPartialCharge(mol)."""
    v = float(Descriptors.MinAbsPartialCharge(_mol_from_smiles(smiles)))
    return f"Min absolute Gasteiger partial charge: {v:.4f}"

# --- EState extremes (electrotopological environment proxies) ---
def get_max_estate_index(smiles: str) -> str:
    """Maximum EState index among atoms. RDKit: Descriptors.MaxEStateIndex(mol)."""
    v = float(Descriptors.MaxEStateIndex(_mol_from_smiles(smiles)))
    return f"Max EState index: {v:.4f}"

def get_min_estate_index(smiles: str) -> str:
    """Minimum EState index among atoms. RDKit: Descriptors.MinEStateIndex(mol)."""
    v = float(Descriptors.MinEStateIndex(_mol_from_smiles(smiles)))
    return f"Min EState index: {v:.4f}"

# --- Aromaticity (atom-level) ---
def get_num_aromatic_atoms(smiles: str) -> str:
    """Number of aromatic atoms. RDKit: sum(atom.GetIsAromatic() for atoms)."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()))
    return f"Aromatic atom count: {v}"

def get_fraction_aromatic_atoms(smiles: str) -> str:
    """Fraction aromatic atoms = (# aromatic atoms)/(# atoms). RDKit aromaticity flags."""
    mol = _mol_from_smiles(smiles)
    n = mol.GetNumAtoms()
    v = (sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / n) if n else 0.0
    return f"Fraction aromatic atoms: {float(v):.4f}"

# --- Formal-charge centers (helps hERG/P-gp/BBB behavior) ---
def get_num_positive_charge_atoms(smiles: str) -> str:
    """Count atoms with formal charge > 0 (cationic centers). RDKit: atom.GetFormalCharge()."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0))
    return f"Cationic center count (formal charge > 0): {v}"

def get_num_negative_charge_atoms(smiles: str) -> str:
    """Count atoms with formal charge < 0 (anionic centers). RDKit: atom.GetFormalCharge()."""
    mol = _mol_from_smiles(smiles)
    v = int(sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0))
    return f"Anionic center count (formal charge < 0): {v}"

# --- Ring type splits (useful for BBB/hERG/CYP) ---
def get_num_aliphatic_rings(smiles: str) -> str:
    """Aliphatic ring count. RDKit: Lipinski.NumAliphaticRings(mol)."""
    v = int(Lipinski.NumAliphaticRings(_mol_from_smiles(smiles)))
    return f"Aliphatic ring count: {v}"

def get_num_saturated_rings(smiles: str) -> str:
    """Saturated ring count. RDKit: Lipinski.NumSaturatedRings(mol)."""
    v = int(Lipinski.NumSaturatedRings(_mol_from_smiles(smiles)))
    return f"Saturated ring count: {v}"

def get_num_heterocycles(smiles: str) -> str:
    """Heterocycle count. RDKit: Lipinski.NumHeterocycles(mol)."""
    v = int(Lipinski.NumHeterocycles(_mol_from_smiles(smiles)))
    return f"Heterocycle count: {v}"

def get_num_aromatic_heterocycles(smiles: str) -> str:
    """Aromatic heterocycle count. RDKit: Lipinski.NumAromaticHeterocycles(mol)."""
    v = int(Lipinski.NumAromaticHeterocycles(_mol_from_smiles(smiles)))
    return f"Aromatic heterocycle count: {v}"

def get_num_aliphatic_heterocycles(smiles: str) -> str:
    """Aliphatic heterocycle count. RDKit: Lipinski.NumAliphaticHeterocycles(mol)."""
    v = int(Lipinski.NumAliphaticHeterocycles(_mol_from_smiles(smiles)))
    return f"Aliphatic heterocycle count: {v}"

def get_num_saturated_heterocycles(smiles: str) -> str:
    """Saturated heterocycle count. RDKit: Lipinski.NumSaturatedHeterocycles(mol)."""
    v = int(Lipinski.NumSaturatedHeterocycles(_mol_from_smiles(smiles)))
    return f"Saturated heterocycle count: {v}"

def get_num_amide_bonds(smiles: str) -> str:
    """Amide bond count (Lipinski). RDKit: Lipinski.NumAmideBonds(mol)."""
    v = int(Lipinski.NumAmideBonds(_mol_from_smiles(smiles)))
    return f"Amide bond count: {v}"

# --- Topological/complexity pack (broad tox panels love these) ---
def get_bertz_ct(smiles: str) -> str:
    """Bertz complexity index. RDKit: Descriptors.BertzCT(mol)."""
    v = float(Descriptors.BertzCT(_mol_from_smiles(smiles)))
    return f"Bertz complexity index: {v:.4f}"

def get_balaban_j(smiles: str) -> str:
    """Balaban J topological index. RDKit: Descriptors.BalabanJ(mol)."""
    v = float(Descriptors.BalabanJ(_mol_from_smiles(smiles)))
    return f"Balaban J index: {v:.4f}"

def get_ipc(smiles: str) -> str:
    """Information content (IPC). RDKit: Descriptors.Ipc(mol)."""
    v = float(Descriptors.Ipc(_mol_from_smiles(smiles)))
    return f"IPC (information content): {v:.4f}"

def get_hall_kier_alpha(smiles: str) -> str:
    """Hall–Kier alpha. RDKit: Descriptors.HallKierAlpha(mol)."""
    v = float(Descriptors.HallKierAlpha(_mol_from_smiles(smiles)))
    return f"Hall–Kier alpha: {v:.4f}"

def get_kappa1(smiles: str) -> str:
    """Kappa1 shape index. RDKit: Descriptors.Kappa1(mol)."""
    v = float(Descriptors.Kappa1(_mol_from_smiles(smiles)))
    return f"Kappa1 shape index: {v:.4f}"

def get_kappa2(smiles: str) -> str:
    """Kappa2 shape index. RDKit: Descriptors.Kappa2(mol)."""
    v = float(Descriptors.Kappa2(_mol_from_smiles(smiles)))
    return f"Kappa2 shape index: {v:.4f}"

def get_kappa3(smiles: str) -> str:
    """Kappa3 shape index. RDKit: Descriptors.Kappa3(mol)."""
    v = float(Descriptors.Kappa3(_mol_from_smiles(smiles)))
    return f"Kappa3 shape index: {v:.4f}"

# --- Stereo (sometimes helps ClinTox/DILI and broad tasks) ---
def get_num_atom_stereo_centers(smiles: str) -> str:
    """Number of atom stereocenters. RDKit: rdMolDescriptors.CalcNumAtomStereoCenters(mol)."""
    v = int(rdMolDescriptors.CalcNumAtomStereoCenters(_mol_from_smiles(smiles)))
    return f"Atom stereocenter count: {v}"

def get_num_unspecified_atom_stereo_centers(smiles: str) -> str:
    """Number of unspecified atom stereocenters. RDKit: rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)."""
    v = int(rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(_mol_from_smiles(smiles)))
    return f"Unspecified atom stereocenter count: {v}"

# ============================================================
# 2) Task-specific packs (add on top of BASIC)
#    (No fragments; relaxed but still interpretable.)
# ============================================================

# ---- hERG (hERG, hERG_Karim, herg_central) ----
RDKIT_HERG_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_aromatic_atoms", "Return number of aromatic atoms."),
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy) using RDKit rdMolDescriptors.CalcLabuteASA."),
    _tool("get_num_amide_bonds", "Return amide bond count using RDKit Lipinski.NumAmideBonds."),
]

# ---- Permeability / absorption (PAMPA, HIA, Bioavailability) ----
RDKIT_PERMEABILITY_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy) using RDKit rdMolDescriptors.CalcLabuteASA."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
]

# ---- BBB (BBB_Martins) ----
RDKIT_BBB_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms (aromatic atoms / total atoms)."),
    _tool("get_num_aromatic_atoms", "Return number of aromatic atoms."),
    _tool("get_num_aliphatic_rings", "Return aliphatic ring count using RDKit Lipinski.NumAliphaticRings."),
    _tool("get_num_saturated_rings", "Return saturated ring count using RDKit Lipinski.NumSaturatedRings."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
]

# ---- P-gp (Pgp_Broccatelli) ----
RDKIT_PGP_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_num_amide_bonds", "Return amide bond count."),
]

# ---- CYP inhibition (CYP*_Veith) ----
RDKIT_CYP_INHIB_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_heterocycles", "Return heterocycle count using RDKit Lipinski.NumHeterocycles."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count using RDKit Lipinski.NumAromaticHeterocycles."),
    _tool("get_num_aliphatic_heterocycles", "Return aliphatic heterocycle count using RDKit Lipinski.NumAliphaticHeterocycles."),
    _tool("get_num_saturated_heterocycles", "Return saturated heterocycle count using RDKit Lipinski.NumSaturatedHeterocycles."),
    _tool("get_num_amide_bonds", "Return amide bond count using RDKit Lipinski.NumAmideBonds."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
]

# ---- CYP substrate (CYP*_Substrate_CarbonMangels) ----
RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_num_heterocycles", "Return heterocycle count."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count."),
    _tool("get_num_amide_bonds", "Return amide bond count."),
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_num_unspecified_atom_stereo_centers", "Return unspecified stereocenter count."),
]

# ---- Genotoxicity / carcinogenicity (AMES, Carcinogens_Lagunin) ----
RDKIT_GENOTOX_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_aromatic_atoms", "Return aromatic atom count."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_balaban_j", "Return Balaban J topological index."),
]

# ---- DILI / ClinTox (systemic tox; broad, tolerant) ----
RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_ipc", "Return IPC (information content)."),
    _tool("get_hall_kier_alpha", "Return Hall–Kier alpha."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
    _tool("get_kappa3", "Return Kappa3 shape index."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_num_unspecified_atom_stereo_centers", "Return unspecified stereocenter count."),
]

# ---- Skin sensitization / Skin Reaction (permeability + reactivity proxies) ----
RDKIT_SKIN_REACTION_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_num_positive_charge_atoms", "Return count of cationic centers (atoms with formal charge > 0)."),
    _tool("get_num_negative_charge_atoms", "Return count of anionic centers (atoms with formal charge < 0)."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
]

# ---- Broad tox panels (Tox21, ToxCast, many “misc tox” sets) ----
RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_max_abs_partial_charge", "Return max absolute Gasteiger partial charge."),
    _tool("get_min_abs_partial_charge", "Return min absolute Gasteiger partial charge."),
    _tool("get_max_estate_index", "Return max EState index among atoms."),
    _tool("get_min_estate_index", "Return min EState index among atoms."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_balaban_j", "Return Balaban J topological index."),
    _tool("get_ipc", "Return IPC (information content)."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
    _tool("get_kappa3", "Return Kappa3 shape index."),
]

# ---- Viral activity (HIV, SARSCoV2 assays): general “binding-ish” chemistry pack ----
RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS: List[Dict[str, Any]] = [
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_fraction_aromatic_atoms", "Return fraction of aromatic atoms."),
    _tool("get_num_heterocycles", "Return heterocycle count."),
    _tool("get_num_aromatic_heterocycles", "Return aromatic heterocycle count."),
    _tool("get_num_atom_stereo_centers", "Return atom stereocenter count."),
    _tool("get_bertz_ct", "Return Bertz complexity index."),
    _tool("get_kappa1", "Return Kappa1 shape index."),
    _tool("get_kappa2", "Return Kappa2 shape index."),
]


# ============================================================
# 3) Per-task mapping (exact keys you listed)
# ============================================================
TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP: Dict[str, List[Dict[str, Any]]] = {
    # hERG family
    "hERG_Karim": RDKIT_HERG_OPENAI_TOOLS,
    "hERG": RDKIT_HERG_OPENAI_TOOLS,
    "herg_central_hERG_inhib": RDKIT_HERG_OPENAI_TOOLS,

    # Genotox / carcinogenicity
    "Carcinogens_Lagunin": RDKIT_GENOTOX_OPENAI_TOOLS,
    "AMES": RDKIT_GENOTOX_OPENAI_TOOLS,

    # Systemic tox
    "DILI": RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS,
    "ClinTox": RDKIT_SYSTEMIC_TOX_OPENAI_TOOLS,

    # Skin sensitization
    "Skin_Reaction": RDKIT_SKIN_REACTION_OPENAI_TOOLS,

    # Broad panels
    "ToxCast": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,
    "Tox21": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,
    "butkiewicz": RDKIT_BROAD_TOX_PANEL_OPENAI_TOOLS,   # (kept broad; name is ambiguous across benchmarks)

    # Permeability / absorption / distribution
    "PAMPA_NCATS": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "HIA_Hou": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "Bioavailability_Ma": RDKIT_PERMEABILITY_OPENAI_TOOLS,
    "BBB_Martins": RDKIT_BBB_OPENAI_TOOLS,
    "Pgp_Broccatelli": RDKIT_PGP_OPENAI_TOOLS,

    # CYP inhibition (Veith)
    "CYP1A2_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2C19_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2C9_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP2D6_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,
    "CYP3A4_Veith": RDKIT_CYP_INHIB_OPENAI_TOOLS,

    # CYP substrate (CarbonMangels)
    "CYP2C9_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,
    "CYP2D6_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,
    "CYP3A4_Substrate_CarbonMangels": RDKIT_CYP_SUBSTRATE_OPENAI_TOOLS,

    # Viral / bioactivity assays
    "HIV": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,
    "SARSCoV2_3CLPro_Diamond": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,
    "SARSCoV2_Vitro_Touret": RDKIT_ANTIVIRAL_ACTIVITY_OPENAI_TOOLS,

    # (If these are not SMILES-based, keep them empty or map to a minimal pack)
    "SAbDab_Chen": [],  # Often antibody/protein-centric; keep empty unless you confirm SMILES inputs exist.
}

# -------------------------
# All RDKit descriptors
# -------------------------
def calc_all_rdkit_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    results = {}
    errors = {}

    # 1) Descriptors._descList（最核心）
    for name, fn in Descriptors._descList:
        try:
            results[name] = fn(mol)
        except Exception as e:
            errors[name] = str(e)

    # 2) rdMolDescriptors: Calc* 系列
    for name in [n for n in dir(rdMolDescriptors) if n.startswith("Calc")]:
        fn = getattr(rdMolDescriptors, name)
        if callable(fn):
            try:
                results[name] = fn(mol)
            except Exception as e:
                errors[name] = str(e)

    # 3) Lipinski: 常见计数
    for name in [n for n in dir(Lipinski) if n.startswith("Num") or n.startswith("Calc")]:
        fn = getattr(Lipinski, name)
        if callable(fn):
            try:
                results[f"Lipinski.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Lipinski.{name}"] = str(e)

    # 4) Crippen
    for name in [n for n in dir(Crippen) if n.startswith("Mol")]:
        fn = getattr(Crippen, name)
        if callable(fn):
            try:
                results[f"Crippen.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Crippen.{name}"] = str(e)

    # 5) GraphDescriptors（有些函数名不是 Calc 开头）
    for name in [n for n in dir(GraphDescriptors) if n and n[0].isupper()]:
        fn = getattr(GraphDescriptors, name)
        if callable(fn):
            try:
                results[f"Graph.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Graph.{name}"] = str(e)

    # 6) Fragments: fr_* 计数
    for name in [n for n in dir(Fragments) if n.startswith("fr_")]:
        fn = getattr(Fragments, name)
        if callable(fn):
            try:
                results[f"Frag.{name}"] = fn(mol)
            except Exception as e:
                errors[f"Frag.{name}"] = str(e)

    return results, errors

if __name__ == "__main__":
    smiles = "CCOC(=O)C(=NOC(C)(C)C(=O)OC(C)(C)C)c1csc(NC(c2ccccc2)(c2ccccc2)c2ccccc2)n1"
    # results, errors = calc_all_rdkit_descriptors(smiles)
    print(RDKIT_OPENAI_TOOLS)
    # print(errors)