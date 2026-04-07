"""
包括了从 Haydn 的 full 文件中获取的 tools, 尤其是和pka相关的
Includes tools obtained from Haydn's full file, especially those related to pKa.
"""
import time

import math
from typing import Dict, Any, List
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
print(f"importing molgpka...")
_t = time.time()
from molgpka import MolGpKa; print(f"from molgpka import MolGpKa: {time.time()-_t:.2f}s")
import json
from .RDKit_tools import _tool


# -------------------------
# Core helpers
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


# Lazy-loaded singleton for MolGpKa predictor
print("Loading MolGpKa predictor...")
_pka_predictor: MolGpKa | None = None
print("MolGpKa predictor loaded.")


def _get_pka_predictor() -> MolGpKa:
    global _pka_predictor
    if _pka_predictor is None:
        _pka_predictor = MolGpKa(uncharged=True)
    return _pka_predictor


def _round4(value: float) -> float:
    return round(value, 4)


def _round_site_pka_map(site_map: Dict[int, float]) -> Dict[int, float]:
    return {atom_map_num: _round4(pka) for atom_map_num, pka in site_map.items()}


# -------------------------
# OpenAI "tools" schema helpers
# -------------------------

def _tool_smiles_only(name: str, description: str) -> Dict[str, Any]:
    """Build an OpenAI-compatible tool schema with a single `smiles` parameter."""
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


def _tool_smiles_and_ph(name: str, description: str) -> Dict[str, Any]:
    """Build an OpenAI-compatible tool schema with `smiles` and optional `ph` parameters."""
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
                    },
                    "ph": {
                        "type": "number",
                        "description": "Target pH (default: 7.4)."
                    }
                },
                "required": ["smiles"],
                "additionalProperties": False
            }
        }
    }


# -------------------------
# Tool implementations
# -------------------------

def predict_pka(smiles: str) -> str:
    """
    Predict pKa values for ionizable sites in a molecule.

    Args:
        smiles (str): The SMILES string of the molecule.

    Returns:
        pKa prediction results including:
            - base_sites: Base-site pKa values (1-indexed atom map numbers).
            - acid_sites: Acid-site pKa values (1-indexed atom map numbers).
            - most_basic_pka: Max base-site pKa (None if no base sites).
            - most_acidic_pka: Min acid-site pKa (None if no acid sites).
            - num_basic_sites: Number of base sites.
            - num_acidic_sites: Number of acid sites.
            - mapped_smiles: SMILES of the protonated molecule with atom map numbers set.
    """
    result = predict_pka_structured(smiles)
    if not result["base_sites"] and not result["acid_sites"]:
        return "No basic or acidic sites predicted."

    output = f"Number of basic sites: {result['num_basic_sites']}\n"
    output += f"Number of acidic sites: {result['num_acidic_sites']}\n"
    output += f"Predicted-site atom-mapped SMILES: {result['mapped_smiles']}\n"

    if result["base_sites"]:
        output += f"Base-site pKa values (atom_map_number: pKa): {result['base_sites']}\n"
        output += f"Most basic pKa: {result['most_basic_pka']:.4f}\n"
    else:
        output += "No base sites predicted.\n"

    if result["acid_sites"]:
        output += f"Acid-site pKa values (atom_map_number: pKa): {result['acid_sites']}\n"
        output += f"Most acidic pKa: {result['most_acidic_pka']:.4f}"
    else:
        output += "No acid sites predicted."

    return output


def estimate_logd(smiles: str, ph: float = 7.4) -> str:
    """
    Estimate logD at a target pH from predicted pKa values and RDKit logP.

    This tool uses a simple Henderson-Hasselbalch approximation to estimate the fraction
    of the neutral (unionized) species at the target pH, then computes:

        logD(pH) ≈ logP + log10(f_neutral)

    where logP is RDKit Wildman-Crippen logP. This is a heuristic intended for
    permeability/BBB-style reasoning and should not be treated as experimental logD.

    For polyprotic molecules, this uses a simple approximation based on the most basic
    and most acidic predicted pKa values (if present).

    Args:
        smiles (str): Query SMILES.
        ph (float): Target pH (default: 7.4).

    Returns:
        Estimated logD plus supporting fields and warnings.
    """
    result = estimate_logd_structured(smiles, ph=ph)
    output = f"Estimated logD at pH {result['ph']:.4f}: {result['logd']:.4f}\n"
    output += f"logP (Wildman-Crippen): {result['logp']:.4f}\n"
    output += f"Fraction neutral at pH {result['ph']:.4f}: {result['fraction_neutral']:.4f}\n"

    if result["warnings"]:
        output += f"Warnings: {'; '.join(result['warnings'])}"

    return output


def predict_pka_structured(smiles: str) -> Dict[str, Any]:
    """
    Structured counterpart of ``predict_pka`` for programmatic use.
    """
    mol = _mol_from_smiles(smiles)
    predictor = _get_pka_predictor()
    prediction = predictor.predict(mol)

    base_sites = prediction.base_sites_1
    acid_sites = prediction.acid_sites_1

    return {
        "base_sites": _round_site_pka_map(base_sites),
        "acid_sites": _round_site_pka_map(acid_sites),
        "most_basic_pka": _round4(max(base_sites.values())) if base_sites else None,
        "most_acidic_pka": _round4(min(acid_sites.values())) if acid_sites else None,
        "num_basic_sites": len(base_sites),
        "num_acidic_sites": len(acid_sites),
        "mapped_smiles": Chem.MolToSmiles(prediction.mol),
    }


def estimate_logd_structured(
    smiles: str,
    ph: float = 7.4,
    pka_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Structured counterpart of ``estimate_logd`` for programmatic use.
    """
    if not (0.0 <= ph <= 14.0):
        raise ValueError(f"pH must be between 0 and 14 (got {ph})")

    mol = _mol_from_smiles(smiles)
    logp = float(Crippen.MolLogP(mol))
    if pka_result is None:
        pka_result = predict_pka_structured(smiles)
    most_basic_pka = pka_result["most_basic_pka"]
    most_acidic_pka = pka_result["most_acidic_pka"]

    warnings: list[str] = []
    if pka_result["num_basic_sites"] > 1:
        warnings.append("Multiple basic sites; using only most_basic_pka for neutral-fraction estimate.")
    if pka_result["num_acidic_sites"] > 1:
        warnings.append("Multiple acidic sites; using only most_acidic_pka for neutral-fraction estimate.")
    if pka_result["base_sites"] and pka_result["acid_sites"]:
        warnings.append("Amphoteric molecule; neutral-fraction estimate assumes independent sites.")

    f_neutral_base = 1.0
    if most_basic_pka is not None:
        f_neutral_base = 1.0 / (1.0 + 10.0 ** (most_basic_pka - ph))

    f_neutral_acid = 1.0
    if most_acidic_pka is not None:
        f_neutral_acid = 1.0 / (1.0 + 10.0 ** (ph - most_acidic_pka))

    f_neutral = f_neutral_base * f_neutral_acid
    if f_neutral <= 0.0:
        warnings.append("Estimated neutral fraction was non-positive; clamping for numerical stability.")
        f_neutral = 1e-12

    f_neutral = min(1.0, max(1e-12, f_neutral))
    if f_neutral < 1e-6:
        warnings.append("Estimated neutral fraction is extremely small; logD estimate may be unreliable.")

    logd = logp + math.log10(f_neutral)
    return {
        "ph": _round4(ph),
        "logp": _round4(logp),
        "logd": _round4(logd),
        "fraction_neutral": _round4(f_neutral),
        "most_basic_pka": most_basic_pka,
        "most_acidic_pka": most_acidic_pka,
        "num_basic_sites": pka_result["num_basic_sites"],
        "num_acidic_sites": pka_result["num_acidic_sites"],
        "warnings": warnings,
    }


# ============================================================
# OpenAI tool definitions
# ============================================================
PKA_TOOL = _tool_smiles_only(
        "predict_pka",
        "Predict pKa values for ionizable sites in a molecule. "
        "Returns base-site and acid-site pKa values (1-indexed atom map numbers), "
        "the most basic/acidic pKa, the number of basic/acidic sites, "
        "and the atom-mapped SMILES."
    )
LOGD_TOOL = _tool_smiles_and_ph(
        "estimate_logd",
        "Estimate logD at a target pH from predicted pKa values and RDKit logP. "
        "Uses a Henderson-Hasselbalch approximation: logD(pH) ≈ logP + log10(f_neutral). "
        "logP is RDKit Wildman-Crippen logP. This is a heuristic intended for helping"
        "your reasoning and should not be treated as experimental logD. "
        "For polyprotic molecules, uses the most basic and most acidic predicted pKa values."
    )

if __name__ == "__main__":
    smiles = 'CC(=O)Oc1ccccc1C(=O)O'  # Aspirin
    print('=== predict_pka ===')
    print(predict_pka(smiles))
    print()
    print('=== estimate_logd (pH=7.4) ===')
    print(estimate_logd(smiles))
    print()
    print('=== estimate_logd (pH=2.0) ===')
    print(estimate_logd(smiles, ph=2.0))
    print()
    print('=== Tool schema ===')
    print(json.dumps([PKA_TOOL, LOGD_TOOL], indent=2, ensure_ascii=False))
