from rdkit import Chem
from rdkit.Chem import Draw
# import selfies as sf
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDrawOptions

# 使用从 OPSIN 获得的 SMILES
smiles = "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1"
# smiles = "[C][=O][N]"
# C[N+](C)(C)CCCC[C@@H](C(=O)[O-])[NH3+]
# smiles = sf.decoder(smiles)
print(smiles)
# 创建分子对象
mol = Chem.MolFromSmiles(smiles)

rdDepictor.SetPreferCoordGen(True)
rdDepictor.Compute2DCoords(mol)

opts = MolDrawOptions()
# opts.reduceOverlap = True

# 绘制分子结构
img = Draw.MolToImage(mol, size=(2000, 2000))
# display(img)
# img.show()
img.save("mol.png")