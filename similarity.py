import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import SDF_dispose
import number
import csv
from tqdm import tqdm
import math


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

def load_pretrained_model(model_path="similar_trained_gcn.pth"):
    checkpoint = torch.load(model_path)
    # 从检查点中获取输出维度，如果没有则使用默认值64
    out_channels = checkpoint.get("out_channels", 64)
    model = ImprovedGCN(in_channels=checkpoint["in_channels"], out_channels=out_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

PRETRAINED_MODEL = load_pretrained_model()

def precompute_all_graphs(active_list):
    """预加载指定分子列表的图数据和name"""
    supplier = Chem.SDMolSupplier('A.sdf')
    graphs = []
    listuseful = []
    mol_names = []  # 存储name
    # print(f"Precomputing {len(active_list)} graphs...")
    for idx in tqdm(active_list):
        try:
            mol_name = "未知"
            mol = supplier[idx]
            if mol is None:
                print(f"{idx} 是无效分子")

            graph = SDF_dispose.molecule_to_pyg_graph(mol)
            # 对齐特征维度
            if graph.x.size(1) < PRETRAINED_MODEL.conv1.in_channels:
                pad_size = PRETRAINED_MODEL.conv1.in_channels - graph.x.size(1)
                graph.x = F.pad(graph.x, (0, pad_size))

            if len(graph.edge_index) == 0:
                print(idx, '是无效分子，其edge_index为空')
                input()
                graph=None
            if mol.HasProp("name"):
                mol_name = mol.GetProp("name")
                if int(mol_name.replace('name', '')) != idx :
                    print(f'{idx}错位')
                else:
                    mol_names.append(mol_name)

            graphs.append(graph)
            listuseful.append(idx)

        except Exception as e:
            print(f"处理分子 {idx} 时出错: {e}")
            # input()
            continue

    return graphs, listuseful, mol_names


def process_batch(model, a_graph, b_graphs, device):
    """处理单个批次，返回a_graph与每个b_graph之间的相似度"""
    # 构建批次
    batch_graphs = [a_graph] + b_graphs
    batch = Batch.from_data_list(batch_graphs).to(device)

    # 计算每个图的节点数，用于后续分离嵌入
    node_slices = []
    start = 0
    for graph in batch_graphs:
        end = start + graph.x.size(0)
        node_slices.append((start, end))
        start = end

    with torch.no_grad():
        # 获取所有节点的嵌入
        all_node_embeddings = model(batch.x, batch.edge_index)

        # 为每个图计算整体嵌入（简单平均池化）
        graph_embeddings = []
        for start, end in node_slices:
            # 提取当前图的节点嵌入并平均
            graph_embed = all_node_embeddings[start:end].mean(dim=0)
            graph_embeddings.append(graph_embed)

    # 转换为张量并计算相似度
    graph_embeddings = torch.stack(graph_embeddings)
    a_embed = graph_embeddings[0].unsqueeze(0)
    b_embeds = graph_embeddings[1:]

    # 归一化嵌入向量
    a_embed = F.normalize(a_embed, dim=1)
    b_embeds = F.normalize(b_embeds, dim=1)

    # 计算余弦相似度
    similarities = torch.mm(a_embed, b_embeds.T).squeeze()
    # print(len(similarities))
    # input()

    # 返回CPU张量的numpy数组
    return similarities.cpu().numpy()
#-------------------------------------
#使用>  <DSSTox_CID>作为名字
#-------------------------------------
# def Mname(number):
#     supplier = Chem.SDMolSupplier('nr-ar.sdf')
#     mol = supplier[number]
#     name = "未知"
#     if mol and mol.HasProp("name"):
#         name = mol.GetProp("name")
#     else:
#         print(f"第 {number} 个分子缺少 <name>属性或分子无效")
#     return name


if __name__ == '__main__':
    # 初始化

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'#强制cpu计算
    model = PRETRAINED_MODEL.to(device)
    a_list =[]
    b_list =[]

    # 获取分子列表
    active_values, active_0_name, active_1_name = number.extract_active_property("A.sdf")
    for name in active_1_name:
        # 从每个字符串提取数字部分
        number = int(name.replace('name', ''))
        a_list.append(number)
    for name in active_0_name:
        # 从每个字符串提取数字部分
        number = int(name.replace('name', ''))
        b_list.append(number)

    # print(a_list)
    # print(b_list)
    # input()

    # 预加载所有图数据和分子名称
    a_graphs, lists0, a_names = precompute_all_graphs(a_list)
    b_graphs, lists1, b_names = precompute_all_graphs(b_list)
    print(lists0)
    print(lists1)



    # 备份数据
    a1, a2, a3 = a_graphs, a_list, a_names
    b1, b2, b3 = b_graphs, b_list, b_names
    # input()

    # 准备结果文件
    with open('nr-ahr_similar(different_active).csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Molecular number1", 'DSSTox_CID_name1', "Molecular number2", 'DSSTox_CID_name2', "similarity"])
        # 遍历每个标记为1的分子
        for idx, (a_idx, a_graph, a_name) in enumerate(zip(a_list, a_graphs, a_names)):
            if idx % 1 == 0:
                print(f"正在处理same1_active: {idx}/{len(a_list)}")

            # 将active_1的分子分批次（每批32个）
            total_b = len(b_list)
            for batch_start in range(0, total_b, 32):
                batch_end = min(batch_start + 32, total_b)
                current_b_graphs = b_graphs[batch_start:batch_end]
                # print(current_b_graphs[0])
                current_b_names = b_names[batch_start:batch_end]  # 获取对应的名称
                # print(current_b_names[0])
                # input()

                # 填充不足32的情况
                if len(current_b_graphs) < 32:
                    padding_count = 32 - len(current_b_graphs)
                    current_b_graphs += [current_b_graphs[-1]] * padding_count
                    current_b_names += [current_b_names[-1]] * padding_count  # 同步填充名称
                # 处理批次
                try:
                    similarities = process_batch(model, a_graph, current_b_graphs, device)
                    # print(len(current_b_graphs))
                    # print(similarities)
                    # print(len(similarities))

                    valid_count = batch_end - batch_start
                    for b_subidx in range(valid_count):
                        b_actual_idx = b_list[batch_start + b_subidx]
                        b_actual_name = b_names[batch_start + b_subidx]  # Use this index for the name

                        writer.writerow([
                            a_idx,
                            a_name,
                            b_actual_idx,
                            b_actual_name,
                            similarities[b_subidx].item()
                        ])

                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
    # ================================================================
    a_graphs, a_list, a_names = b1, b2, b3  # 使用标记为0的分子
    b_graphs, b_list, b_names = b1, b2, b3  # 使用标记为0的分子

    with open('nr-ahr_similar(same0_active).csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Molecular number1", 'DSSTox_CID_name1', "Molecular number2", 'DSSTox_CID_name2', "similarity"])
        # 遍历每个标记为1的分子
        for idx, (a_idx, a_graph, a_name) in enumerate(zip(a_list, a_graphs, a_names)):
            if idx % 1 == 0:
                print(f"正在处理same1_active: {idx}/{len(a_list)}")

            # 将active_1的分子分批次（每批32个）
            total_b = len(b_list)
            for batch_start in range(0, total_b, 32):
                batch_end = min(batch_start + 32, total_b)
                current_b_graphs = b_graphs[batch_start:batch_end]
                # print(current_b_graphs[0])
                current_b_names = b_names[batch_start:batch_end]  # 获取对应的名称
                # print(current_b_names[0])
                # input()

                # 填充不足32的情况
                if len(current_b_graphs) < 32:
                    padding_count = 32 - len(current_b_graphs)
                    current_b_graphs += [current_b_graphs[-1]] * padding_count
                    current_b_names += [current_b_names[-1]] * padding_count  # 同步填充名称
                # 处理批次
                try:
                    similarities = process_batch(model, a_graph, current_b_graphs, device)
                    # print(len(current_b_graphs))
                    # print(similarities)
                    # print(len(similarities))

                    valid_count = batch_end - batch_start
                    for b_subidx in range(valid_count):
                        b_actual_idx = b_list[batch_start + b_subidx]
                        b_actual_name = b_names[batch_start + b_subidx]  # Use this index for the name

                        writer.writerow([
                            a_idx,
                            a_name,
                            b_actual_idx,
                            b_actual_name,
                            similarities[b_subidx].item()
                        ])

                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue
    # #==============================================================================
    a_graphs, a_list, a_names = a1, a2, a3  # 使用标记为1的分子
    b_graphs, b_list, b_names = a1, a2, a3  # 使用标记为1的分子

    with open('nr-ahr_similar(same1_active).csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Molecular number1", 'DSSTox_CID_name1', "Molecular number2", 'DSSTox_CID_name2', "similarity"])
        # 遍历每个标记为1的分子
        for idx, (a_idx, a_graph, a_name) in enumerate(zip(a_list, a_graphs, a_names)):
            if idx % 1 == 0:
                print(f"正在处理same1_active: {idx}/{len(a_list)}")

            # 将active_1的分子分批次（每批32个）
            total_b = len(b_list)
            for batch_start in range(0, total_b, 32):
                batch_end = min(batch_start + 32, total_b)
                current_b_graphs = b_graphs[batch_start:batch_end]
                # print(current_b_graphs[0])
                current_b_names = b_names[batch_start:batch_end]  # 获取对应的名称
                # print(current_b_names[0])
                # input()

                # 填充不足32的情况
                if len(current_b_graphs) < 32:
                    padding_count = 32 - len(current_b_graphs)
                    current_b_graphs += [current_b_graphs[-1]] * padding_count
                    current_b_names += [current_b_names[-1]] * padding_count  # 同步填充名称
                # 处理批次
                try:
                    similarities = process_batch(model, a_graph, current_b_graphs, device)
                    # print(len(current_b_graphs))
                    # print(similarities)
                    # print(len(similarities))

                    valid_count = batch_end - batch_start
                    for b_subidx in range(valid_count):
                        b_actual_idx = b_list[batch_start + b_subidx]
                        b_actual_name = b_names[batch_start + b_subidx]  # Use this index for the name

                        writer.writerow([
                            a_idx,
                            a_name,
                            b_actual_idx,
                            b_actual_name,
                            similarities[b_subidx].item()
                        ])

                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    continue



    print("计算完成！结果已保存到nr-ahr_similar(different_active).csv")