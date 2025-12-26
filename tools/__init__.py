from .AccFG import *
from .RDKit_tools import *

TOOLS = RDKIT_OPENAI_TOOLS + AccFG_OPENAI_TOOLS  # 因为 AccFG_OPENAI_TOOLS 里面的 name 和实际的调用的函数不一致所以注意下面 tool_map 的映射

def get_function_by_name(name):
    tool_map = {
        "describe_high_level_fg_fragments": cached_describe_high_level_fg_fragments,  # 如果有缓存的直接读缓存的结果省时间 / read from cache if available to save time
        "get_molecular_weight": get_molecular_weight,
        "get_exact_molecular_weight": get_exact_molecular_weight,
        "get_heavy_atom_count": get_heavy_atom_count,
        "get_mol_logp": get_mol_logp,
        "get_tpsa": get_tpsa,
        "get_hbd": get_hbd,
        "get_hba": get_hba,
        "get_num_rotatable_bonds": get_num_rotatable_bonds,
        "get_fraction_csp3": get_fraction_csp3,
        "get_labute_asa": get_labute_asa,
        "get_mol_mr": get_mol_mr,
    }
    return tool_map.get(name)