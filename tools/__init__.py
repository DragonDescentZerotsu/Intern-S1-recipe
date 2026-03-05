from .AccFG import *
from .RDKit_tools import *
from .ePSA_3D import get_3d_exposed_polar_surface, SASA_OPENAI_TOOLS
from .pka_related_tools import *
from .standardize_tools import *
from .arithmetic import evaluate_arithmetic, ARITHMETIC_OPENAI_TOOLS

BASIC_TOOLS = RDKIT_BASIC_OPENAI_TOOLS + AccFG_OPENAI_TOOLS + [PKA_TOOL, LOGD_TOOL] + STANDARDIZE_OPENAI_TOOLS + SASA_OPENAI_TOOLS + ARITHMETIC_OPENAI_TOOLS # 因为 AccFG_OPENAI_TOOLS 里面的 name 和实际的调用的函数不一致所以注意下面 tool_map 的映射

def get_function_by_name(name):
    tool_map = {
        "describe_high_level_fg_fragments": cached_describe_high_level_fg_fragments,  # 如果有缓存的直接读缓存的结果省时间 / read from cache if available to save time
        "get_molecular_weight": get_molecular_weight,
        "get_exact_molecular_weight": get_exact_molecular_weight,
        "get_heavy_atom_count": get_heavy_atom_count,

        "get_mol_logp": get_mol_logp,
        "get_logp": get_mol_logp,
        "get_logP": get_mol_logp,

        "get_tpsa": get_tpsa,
        "get_hbd": get_hbd,
        "get_hba": get_hba,
        "get_num_rotatable_bonds": get_num_rotatable_bonds,
        "get_fraction_csp3": get_fraction_csp3,
        "get_labute_asa": get_labute_asa,
        "get_mol_mr": get_mol_mr,
        "get_ring_count": get_ring_count,
        "get_num_aromatic_rings": get_num_aromatic_rings,
        "get_formal_charge": get_formal_charge,
        "get_qed": get_qed,
        "get_num_heteroatoms": get_num_heteroatoms,
        "get_max_abs_partial_charge": get_max_abs_partial_charge,
        "get_min_abs_partial_charge": get_min_abs_partial_charge,
        "get_max_estate_index": get_max_estate_index,
        "get_min_estate_index": get_min_estate_index,
        "get_num_aromatic_atoms": get_num_aromatic_atoms,
        "get_fraction_aromatic_atoms": get_fraction_aromatic_atoms,
        "get_num_positive_charge_atoms": get_num_positive_charge_atoms,
        "get_num_negative_charge_atoms": get_num_negative_charge_atoms,
        "get_num_aliphatic_rings": get_num_aliphatic_rings,
        "get_num_saturated_rings": get_num_saturated_rings,
        "get_num_heterocycles": get_num_heterocycles,
        "get_num_aromatic_heterocycles": get_num_aromatic_heterocycles,
        "get_num_aliphatic_heterocycles": get_num_aliphatic_heterocycles,
        "get_num_saturated_heterocycles": get_num_saturated_heterocycles,
        "get_num_amide_bonds": get_num_amide_bonds,
        "get_bertz_ct": get_bertz_ct,
        "get_balaban_j": get_balaban_j,
        "get_ipc": get_ipc,
        "get_hall_kier_alpha": get_hall_kier_alpha,
        "get_kappa1": get_kappa1,
        "get_kappa2": get_kappa2,
        "get_kappa3": get_kappa3,
        "get_num_atom_stereo_centers": get_num_atom_stereo_centers,
        "get_num_unspecified_atom_stereo_centers": get_num_unspecified_atom_stereo_centers,
        "get_3d_exposed_polar_surface": get_3d_exposed_polar_surface,

        "predict_pka": predict_pka,
        "predict_pka": predict_pka,

        "estimate_logd": estimate_logd,
        "estimate_logD": estimate_logd,

        "remove_salts": remove_salts,
        "match_substructure": match_substructure,
        "analyze_ring_systems": analyze_ring_systems,
        "classify_ionization": classify_ionization,
        "score_structural_alerts": score_structural_alerts,
        "extract_pharmacophore_features": extract_pharmacophore_features,
        "get_esol": get_esol,
        "evaluate_arithmetic": evaluate_arithmetic,
    }
    return tool_map.get(name)