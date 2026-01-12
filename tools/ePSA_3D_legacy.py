from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem
import math
import re
import tempfile
import os
from .RDKit_tools import _tool
import freesasa
freesasa.setVerbosity(freesasa.nowarnings)


def rdkit_embed_minimize(smiles: str, n_confs: int = 50, seed: int = 0):
    """
    生成多个构象，并用 MMFF94s 最小化，返回：
    - mol: 含 H、含多个构象的 RDKit Mol
    - energies: [(confId, energy), ...] 按能量从低到高排序

    兜底策略：
    - 若 MMFF 参数失败，则 fallback 用 UFF 做最小化（但能量不可与 MMFF 混用比较）
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.numThreads = 0
    params.pruneRmsThresh = 0.25  # 去重：避免大量重复构象
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    if not conf_ids:
        raise RuntimeError("Conformer embedding failed")

    # 尝试 MMFF
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    energies = []

    if props is not None:
        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=500)
            energies.append((cid, float(ff.CalcEnergy())))
    else:
        # fallback UFF
        for cid in conf_ids:
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if ff is None:
                continue
            ff.Minimize(maxIts=500)
            energies.append((cid, float(ff.CalcEnergy())))

    if not energies:
        raise RuntimeError("Forcefield minimization failed for all conformers")

    energies.sort(key=lambda x: x[1])
    return mol, energies

def _get_atom_areas(result, n_atoms: int):
    # 新版：atomAreas() -> list
    if hasattr(result, "atomAreas"):
        return list(result.atomAreas())
    # 旧版：atomArea(i) -> float
    if hasattr(result, "atomArea"):
        return [float(result.atomArea(i)) for i in range(n_atoms)]
    raise AttributeError("FreeSASA Result has neither atomAreas() nor atomArea(i)")


def _parse_pdb_elements_in_order(pdb_block: str) -> list[str]:
    """
    兜底：从 PDB 文本中按 ATOM/HETATM 顺序解析元素符号。
    优先使用 element 字段(列 77-78)，没有则从 atom name 推断。
    """
    elems = []
    for line in pdb_block.splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        elem = line[76:78].strip()
        if elem:
            elems.append(elem.capitalize())
            continue

        # fallback: 从 atom name 推断（处理 1CL / CL1 / Br 等情况）
        name = line[12:16].strip()
        name = name.lstrip("0123456789")
        m = re.match(r"^([A-Za-z]{1,2})", name)
        if not m:
            elems.append("")
        else:
            elems.append(m.group(1).capitalize())
    return elems


def exposed_polar_sasa_for_conf(
    mol: Chem.Mol,
    confId: int,
    probe_radius: float = 1.4,
    polar_symbols: set[str] | None = None,
    include_charged: bool = True,
):
    if polar_symbols is None:
        polar_symbols = {"O", "N", "S", "P"}

    pdb = Chem.MolToPDBBlock(mol, confId=confId)

    # freesasa 参数：可控 probe 半径
    fs_params = freesasa.Parameters()
    fs_params.setProbeRadius(float(probe_radius))

    # ✅ 关键：freesasa.Structure 只吃“文件名”，不吃“pdb字符串”
    # ✅ 同时打开 hetatm/hydrogen，否则它会默认跳过 HETATM 和 H
    fs_options = {
        "hetatm": True,
        "hydrogen": True,
        "skip-unknown": False,
        "halt-at-unknown": False,
    }

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb)
            tmp_path = f.name

        structure = freesasa.Structure(tmp_path, options=fs_options)
        result = freesasa.calc(structure, fs_params)
        n_fs = structure.nAtoms()
        atom_areas = _get_atom_areas(result, n_fs)

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    n_rd = mol.GetNumAtoms()
    n_fs = structure.nAtoms()

    if len(atom_areas) != n_fs:
        raise RuntimeError("freesasa atomAreas length mismatch with freesasa structure")

    total = float(sum(atom_areas))

    # 理想情况：freesasa 读到的原子数 == RDKit 原子数（因为我们开启了 hetatm + hydrogen）
    if n_fs == n_rd:
        polar = 0.0
        for i in range(n_rd):
            a = float(atom_areas[i])
            atom = mol.GetAtomWithIdx(i)
            sym = atom.GetSymbol()
            if sym in polar_symbols:
                polar += a
            elif include_charged and atom.GetFormalCharge() != 0:
                polar += a
        return polar, total

    # 如果仍不相等（少见），给出明确报错提示你下一步怎么查
    raise RuntimeError(
        f"Atom count mismatch: freesasa nAtoms={n_fs}, RDKit nAtoms={n_rd}. "
        "Check freesasa options or whether some atoms were skipped."
    )



def exposed_polar_sasa_ensemble(
    smiles: str,
    n_confs: int = 80,
    top_k: int = 10,
    probe_radius: float = 1.4,
    boltzmann_T: float | None = 298.15,
):
    """
    生成构象 -> 最小化 -> 取 top_k 低能构象 -> 计算每个构象的 ePSA/totalSASA/polar_fraction
    可选：返回 Boltzmann 加权平均（仅当使用同一力场能量、且能量单位可比时更合理）
    """
    mol, energies = rdkit_embed_minimize(smiles, n_confs=n_confs)
    top = energies[: max(1, int(top_k))]

    rows = []
    for cid, e in top:
        polar, total = exposed_polar_sasa_for_conf(mol, cid, probe_radius=probe_radius)
        rows.append(
            {
                "confId": cid,
                "E": e,
                "polar_sasa": polar,
                "total_sasa": total,
                "polar_fraction": (polar / total) if total > 0 else float("nan"),
            }
        )

    # 统计：min/median/mean
    polar_vals = [r["polar_sasa"] for r in rows]
    frac_vals = [r["polar_fraction"] for r in rows]

    stats = {
        "polar_sasa_min": min(polar_vals),
        "polar_sasa_mean": sum(polar_vals) / len(polar_vals),
        "polar_fraction_min": min(frac_vals),
        "polar_fraction_mean": sum(frac_vals) / len(frac_vals),
    }

    # Boltzmann 加权（可选）
    boltz = None
    if boltzmann_T is not None and len(rows) >= 2:
        # RDKit MMFF/UFF 的能量通常可以当作 kcal/mol 量级做相对权重（近似）
        # 权重 w_i = exp(-(E_i - Emin)/(R*T)), R=0.001987 kcal/mol/K
        R = 0.0019872041
        Emin = rows[0]["E"]
        ws = [math.exp(-(r["E"] - Emin) / (R * boltzmann_T)) for r in rows]
        Z = sum(ws)
        ws = [w / Z for w in ws]
        boltz = {
            "polar_sasa_boltz": sum(w * r["polar_sasa"] for w, r in zip(ws, rows)),
            "polar_fraction_boltz": sum(w * r["polar_fraction"] for w, r in zip(ws, rows)),
        }

    return rows, stats, boltz


if __name__ == "__main__":
    smiles = "C[C@]12C[C@H]([C@@H]([C@@]1(CC(=O)[C@@]3([C@H]2CC=C4[C@H]3C=C(C(=O)C4(C)C)O)C)C)[C@](C)(C(=O)/C=C\\C(C)(C)O)O)O"  # 你的 SMILES
    rows, stats, boltz = exposed_polar_sasa_ensemble(
        smiles,
        n_confs=120,
        top_k=15,
        probe_radius=1.4,
        boltzmann_T=298.15,
    )

    for r in rows:
        print(r)
    print("stats:", stats)
    print("boltz:", boltz)
