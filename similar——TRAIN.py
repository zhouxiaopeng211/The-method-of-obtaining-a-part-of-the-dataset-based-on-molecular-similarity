# train_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import SDF_dispose
import random  # 导入随机模块
import numpy as np
from tqdm import tqdm


# ----------------------------
# 1. 定义GCN模型
# ----------------------------
class ImprovedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=64, dropout=0.2):
        super(ImprovedGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, out_channels)
        self.dropout = dropout
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden_channels * 2)

    def forward(self, x, edge_index):
        # 第一层卷积
        x = self.conv1(x, edge_index)
        if x.size(0) > 1:  # 只有在批量大于1时才能应用批归一化
            x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)
        if x.size(0) > 1:
            x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第三层卷积
        x = self.conv3(x, edge_index)
        return x


# ----------------------------
# 2. 数据加载与预处理
# ----------------------------
def load_molecules(sdf_path):
    """加载分子，不进行过滤"""
    supplier = Chem.SDMolSupplier(sdf_path)
    molecules = []
    for idx, mol in enumerate(supplier):
        if mol is not None:
            molecules.append((idx, mol))  # 保存索引和分子对象
    return molecules


def align_feature_dimension(graphs):
    """统一所有图的特征维度"""
    max_dim = max(g.x.size(1) for g in graphs)
    aligned_graphs = []
    for g in graphs:
        current_dim = g.x.size(1)
        if current_dim < max_dim:
            g.x = F.pad(g.x, (0, max_dim - current_dim))
        elif current_dim > max_dim:
            g.x = g.x[:, :max_dim]
        aligned_graphs.append(g)
    return aligned_graphs, max_dim


def calculate_tanimoto(mol1, mol2):
    """计算两个分子之间的谷本系数（Tanimoto coefficient）"""
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def create_training_pairs(molecules, num_pairs=1000):
    """创建训练对，使用谷本系数作为相似性标准"""
    mol_objects = [mol for _, mol in molecules]
    train_data = []
    mol_count = len(mol_objects)

    # 随机选择分子对进行训练
    pairs_to_generate = min(num_pairs, mol_count * (mol_count - 1) // 2)

    print(f"生成 {pairs_to_generate} 个训练对...")

    selected_pairs = set()
    pbar = tqdm(total=pairs_to_generate)

    while len(train_data) < pairs_to_generate:
        i, j = random.sample(range(mol_count), 2)
        pair_key = tuple(sorted([i, j]))

        # 避免重复对
        if pair_key in selected_pairs:
            continue

        selected_pairs.add(pair_key)

        # 计算谷本系数作为相似性目标
        tanimoto = calculate_tanimoto(mol_objects[i], mol_objects[j])

        # 添加到训练数据
        train_data.append((i, j, tanimoto))
        pbar.update(1)

    pbar.close()
    return train_data, mol_objects


# -----------------------------
def train_model(sdf_path="A.sdf", save_path="similar_trained_gcn.pth"):
    # 加载分子及其原始索引
    molecules = load_molecules(sdf_path)
    if len(molecules) < 2:
        raise ValueError("有效分子不足，至少需要2个分子进行训练")

    print(f"加载的分子总数: {len(molecules)}")

    # 创建训练对，使用谷本系数作为相似性标准
    train_pairs, mol_objects = create_training_pairs(molecules, num_pairs=2000)

    # 转换为图数据结构
    graphs = []
    for idx, mol in molecules:
        try:
            graph = SDF_dispose.molecule_to_pyg_graph(mol)
            if graph.x.size(0) > 0 and graph.edge_index.size(1) > 0:
                graphs.append(graph)
            else:
                print(f"分子 {idx} 转换为空图")
        except Exception as e:
            print(f"分子 {idx} 转换失败: {e}")

    print(f"成功转换为图的分子数: {len(graphs)}")

    # 统一特征维度
    aligned_graphs, in_channels = align_feature_dimension(graphs)
    print(f"输入特征维度: {in_channels}")

    # 初始化改进版GCN模型
    model = ImprovedGCN(in_channels, hidden_channels=32, out_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练循环
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 5

    for epoch in range(50):
        total_loss = 0
        batch_size = 0

        # 打乱训练对
        random.shuffle(train_pairs)

        # 执行训练
        for idx1, idx2, target in train_pairs:
            # 确保索引有效
            if idx1 >= len(aligned_graphs) or idx2 >= len(aligned_graphs):
                continue

            optimizer.zero_grad()

            # 获取嵌入
            g1 = aligned_graphs[idx1]
            g2 = aligned_graphs[idx2]

            emb1 = model(g1.x, g1.edge_index)
            emb2 = model(g2.x, g2.edge_index)

            # 全局平均池化
            vec1 = torch.mean(emb1, dim=0)
            vec2 = torch.mean(emb2, dim=0)

            # 归一化向量
            vec1 = F.normalize(vec1, dim=0)
            vec2 = F.normalize(vec2, dim=0)

            # 计算余弦相似度
            similarity = torch.dot(vec1, vec2)

            # 使用平滑L1损失
            target_tensor = torch.tensor(target, dtype=torch.float)
            loss = F.smooth_l1_loss(similarity, target_tensor)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_size += 1

        # 计算平均损失
        avg_loss = total_loss / batch_size if batch_size > 0 else float('inf')
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 学习率调度
        scheduler.step(avg_loss)

        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                "state_dict": model.state_dict(),
                "in_channels": in_channels,
                "out_channels": 64  # 添加输出维度信息
            }, save_path)
            print(f"模型已保存至 {save_path}，当前最佳loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"早停触发，{max_patience}个epoch无改善")
                break

    return model


# 主程序入口
if __name__ == "__main__":
    train_model()
