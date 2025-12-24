from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Lipinski, Crippen, GraphDescriptors, Fragments
from typing import Dict, Any


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


# -------------------------
# Exposure / permeability proxies (SMILES-only)
# -------------------------
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

def get_labute_asa(smiles: str) -> str:
    """
    Labute's Approximate Surface Area (ASA proxy).
    Computed from fragments/topology (not a true solvent-accessible 3D surface).
    """
    return f"Labute's Approximate Surface Area (ASA proxy): {float(rdMolDescriptors.CalcLabuteASA(_mol_from_smiles(smiles))):.4f}"

def get_mol_mr(smiles: str) -> str:
    """
    Wildman–Crippen molar refractivity (MR) estimate.
    Proxy for polarizability/volume; fragment-based (2D).
    """
    return f"Wildman–Crippen molar refractivity (MR) estimate: {float(Crippen.MolMR(_mol_from_smiles(smiles))):.4f}"

RDKIT_OPENAI_TOOLS = [
    _tool("get_molecular_weight", "Return the average molecular weight (Daltons)."),
    _tool("get_exact_molecular_weight", "Return the monoisotopic exact molecular weight (Daltons)."),
    _tool("get_heavy_atom_count", "Return the number of non-hydrogen atoms (heavy atoms)."),
    _tool("get_mol_logp", "Return Wildman–Crippen cLogP (octanol/water)."),
    _tool("get_tpsa", "Return topological polar surface area TPSA (Å^2)."),
    _tool("get_hbd", "Return Lipinski H-bond donors count."),
    _tool("get_hba", "Return Lipinski H-bond acceptors count."),
    _tool("get_num_rotatable_bonds", "Return rotatable bonds count."),
    _tool("get_fraction_csp3", "Return FractionCSP3 = (# sp3 carbon atoms)/(# total carbon atoms), Often used as a simple “3D character / saturation” proxy from 2D structure."),
    _tool("get_labute_asa", "Return Labute approximate surface area (ASA proxy)."),
    _tool("get_mol_mr", "Return Wildman–Crippen molar refractivity (MR)."),
]

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