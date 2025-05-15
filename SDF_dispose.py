from rdkit import Chem
from torch_geometric.data import Data
import torch
#-----------------------------------------------------------------------------------------------------------------------
#turn to smile
def sdf_to_smiles(sdf_path):
    # 读取 SDF 文件
    supplier = Chem.SDMolSupplier(sdf_path)
    smiles_list = []

    # 遍历所有分子
    for mol in supplier:
        if mol is not None:  # 跳过无效分子
            # 生成标准 SMILES（默认规范化）
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
        else:
            print("Warning: 跳过无法解析的分子")

    return smiles_list

#-----------------------------------------------------------------------------------------------------------------------
#turn to graph


def molecule_to_pyg_graph(mol):
    # 节点特征（例如原子类型编码）
    atom_features = []
    for atom in mol.GetAtoms():
        feature = [
            atom.GetAtomicNum(),  # 原子序数
            atom.GetDegree(),  # 连接度
            atom.GetHybridization().real  # 杂化类型
        ]
        atom_features.append(feature)
    x = torch.tensor(atom_features, dtype=torch.float)

    # 边索引和边特征
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        # 添加双向边（无向图）
        edge_index.extend([[start, end], [end, start]])
        # 边特征（例如键类型）
        bond_feature = [
            bond.GetBondTypeAsDouble(),
            bond.IsInRing()
        ]
        edge_attr.extend([bond_feature, bond_feature])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)





