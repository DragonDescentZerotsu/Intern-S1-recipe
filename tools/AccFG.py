'''
This script is used to use AccFG to parse SMILES string into functional groups (or with attachment points).
'''

from rdkit import RDConfig
import os, csv, re
from accfg import AccFG, draw_mol_with_fgs, molimg
from accfg.draw import print_fg_tree
from rdkit import Chem


def fg_fragment_with_attachment_points(smiles: str, fg_atoms: tuple[int, ...]):
    mol = Chem.MolFromSmiles(smiles)
    fg = set(fg_atoms)

    # 1) 找到所有“跨边界”的 bond（一个端点在 fg，另一个不在）
    cut_bond_ids = []
    cut_bonds = []
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if (a1 in fg) ^ (a2 in fg):
            cut_bond_ids.append(b.GetIdx())
            cut_bonds.append((a1, a2))

    # 2) 断键并在断点加 dummy 原子（*）
    # FragmentOnBonds 会返回“一个整体 mol”，里面包含所有碎片，并用 dummy 标记断点  [oai_citation:3‡rdkit.readthedocs.io](https://rdkit.readthedocs.io/en/latest/GettingStartedInPython.html)

    if not cut_bond_ids:
        print(f"Warning: No cut bonds found for {smiles}, fg_atoms: {fg_atoms}")
        return None, None, None
    frag_mol = Chem.FragmentOnBonds(mol, cut_bond_ids, addDummies=True)

    # 3) 把整体 mol 拆成各个连通分量
    # GetMolFrags(asMols=False) 会给每个碎片的 atom id 列表；asMols=True 则直接返回 Mol 对象  [oai_citation:4‡buildmedia.readthedocs.org](https://buildmedia.readthedocs.org/media/pdf/rdkit/latest/rdkit.pdf?utm_source=chatgpt.com)
    frags_atom_ids = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)

    # 4) 选出“和 fg 重叠最多”的那个碎片（通常就是你的功能团那块 + dummy）
    best_ids = max(frags_atom_ids, key=lambda ids: len(fg.intersection(ids)))

    # 5) 输出该碎片的 SMILES（会包含 dummy 连接点）
    frag_smiles = Chem.MolFragmentToSmiles(
        frag_mol,
        atomsToUse=list(best_ids),
        isomericSmiles=True
    )

    # （可选）也可以看看“断开后的整分子”长啥样，通常是多个组分用 '.' 分开
    all_fragged_smiles = Chem.MolToSmiles(frag_mol, isomericSmiles=True)

    return frag_smiles, all_fragged_smiles, cut_bonds


def describe_high_level_fg_fragments_with_attachment_points(smiles:str):
    '''
    Parse SMILES string into functional groups with attachment points
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES and attachment points
    '''
    afg = AccFG(print_load_info=False)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles, _, _ = fg_fragment_with_attachment_points(smiles, FG_atom_ids)
            if FG_fragment_smiles is None:
                FG_fragment_smiles_list.append('Not matched')
                continue
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside <SMILES>{smiles}</SMILES> are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            if FG_fragment_smiles == 'Not matched':
                FGs_description += " Not matched, "
                continue
            FGs_description += f" <SMILES>{FG_fragment_smiles}</SMILES>, "
        FGs_description += "\n"
    return FGs_description


def describe_high_level_fg_fragments(smiles:str):
    '''
    Parse SMILES string into functional groups
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES
    '''
    afg = AccFG(print_load_info=False)
    mol = Chem.MolFromSmiles(smiles)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,   # 不强制重排，通常更便于对照
            )
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside <SMILES>{smiles}</SMILES> are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            # if FG_fragment_smiles == 'Not matched':
            #     FGs_description += " Not matched, "
            #     continue
            FGs_description += f" <SMILES>{FG_fragment_smiles}</SMILES>, "
        FGs_description += "\n"
    return FGs_description

def describe_high_level_fg_fragments_no_special_token(smiles:str):
    '''
    Parse SMILES string into functional groups
    Args:
        smiles (str): SMILES of the molecule
    Returns:
        FGs_description (str): description of founctional groups in the molecule with fragment SMILES
    '''
    afg = AccFG(print_load_info=False)
    mol = Chem.MolFromSmiles(smiles)

    # show_atoms=True  -> 让输出带上“功能团命中的原子编号”
    # show_graph=True  -> 返回 fg_graph（用于打印树/结构化组织）
    fgs, fg_graph = afg.run(smiles, show_atoms=True, show_graph=True)

    FG_name_SMILES_fragment_map = {}
    
    for FG_name, FG_atom_ids_list in fgs.items():
        FG_fragment_smiles_list = []
        for FG_atom_ids in FG_atom_ids_list:
            FG_fragment_smiles = Chem.MolFragmentToSmiles(
                mol,
                atomsToUse=list(FG_atom_ids),
                isomericSmiles=True,
                canonical=False,   # 不强制重排，通常更便于对照
            )
            FG_fragment_smiles_list.append(FG_fragment_smiles)
        
        FG_name_SMILES_fragment_map[FG_name] = FG_fragment_smiles_list

    FGs_description = f"The functional groups inside {smiles} are:\n"
    for i, (FG_name, FG_fragment_smiles_list) in enumerate(FG_name_SMILES_fragment_map.items()):
        FGs_description += f"{i+1}. {FG_name}:"
        FGs_description += f"\n   Count:{len(FG_fragment_smiles_list)}"
        FGs_description += "\n   Corresponding fragment SMILES:"
        for FG_fragment_smiles in FG_fragment_smiles_list:
            # if FG_fragment_smiles == 'Not matched':
            #     FGs_description += " Not matched, "
            #     continue
            FGs_description += f" {FG_fragment_smiles}, "  # Difference here, no <SMILES> token
        FGs_description += "\n"
    return FGs_description


if __name__ == "__main__":
    # example usage
    smiles = "O=C(O)[C@H](Cc1cnc[nH]1)N1C(=O)c2ccccc2C1=O"
    print(describe_high_level_fg_fragments_with_attachment_points(smiles))
    print('\n')
    print(describe_high_level_fg_fragments(smiles))
    print('\n')
    print(describe_high_level_fg_fragments_no_special_token(smiles))
