import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import re
import random
import number
import similarity
import copy
import matrix_make
import torch.optim as opt
import torch as T
import os
import numpy as np
from torch_geometric.nn import GATConv
from torch.nn import Linear
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def extract_digits(s):
    # 把开头的所有非数字字符替换为空
    return int(re.sub(r'^\D+', '', s))

class GNN(nn.Module):
    def __init__(self, INPUT):
        super(GNN, self).__init__()
        # Building INPUT
        self.INPUT = INPUT
        # Defining base variables
        self.CHECKPOINT_DIR = self.INPUT["CHECKPOINT_DIR"]
        self.CHECKPOINT_FILE = os.path.join(self.CHECKPOINT_DIR, self.INPUT["NAME"])
        self.SIZE_LAYERS = self.INPUT["SIZE_LAYERS"]
        self.initial_conv = GATConv(self.SIZE_LAYERS[0], self.SIZE_LAYERS[1])
        self.conv1 = GATConv(self.SIZE_LAYERS[1], self.SIZE_LAYERS[2])
        self.conv2 = GATConv(self.SIZE_LAYERS[2], self.SIZE_LAYERS[2])
        self.linear = Linear(self.SIZE_LAYERS[2], self.SIZE_LAYERS[3])
        self.optimizer = opt.Adam(self.parameters(), lr=self.INPUT["LR"])
        self.criterion = nn.MSELoss()

    def forward(self, x, edge_index):  # forward propagation includes defining layers
        out = F.relu(self.initial_conv(x, edge_index=edge_index))
        out = F.relu(self.conv1(out, edge_index=edge_index))
        out = F.relu(self.conv2(out, edge_index=edge_index))
        return self.linear(out)

    def save_checkpoint(self):
        print(f'保存模型到 {self.CHECKPOINT_FILE}...')
        T.save(self.state_dict(), self.CHECKPOINT_FILE)

    def load_checkpoint(self):
        print(f'从 {self.CHECKPOINT_FILE} 加载模型...')
        self.load_state_dict(T.load(self.CHECKPOINT_FILE))

def listmake(a):

    matrix_list = matrix[a, :]
    name = []
    active1 = []
    active0 = []
    for i in range(0, len(matrix_list)):
        if matrix_list[i] != 0:
            name.append(matrix_list[i])
    # print(name)
    # print(len(name))
    for s in name:
        # 匹配模式：符号(+/-) + 固定名称"name" + 末尾数字
        match = re.match(r'^([+-])name(\d+)$', s)  # 直接匹配固定名称"name"
        if match:
            sign = match.group(1)  # 提取符号
            number = match.group(2)  # 提取数字DD
            if sign == '-':
                active0.append(int(number))
            elif sign == '+':
                active1.append(int(number))
            # print(f"符号: {sign}, 名称: name, 数字: {number}")
        else:
            print(f"格式错误: {s}")
    # print(len(active1), len(active0))
    # print(active1, active0)
    a = active_1_name[a]
    anchor = extract_digits(a)

    # print('\n',active1,'\n',active0,'\n',anchor)
    g1, _, _ = similarity.precompute_all_graphs(active1)
    g0, _, _ = similarity.precompute_all_graphs(active0)
    g2, _, _ = similarity.precompute_all_graphs([anchor])



    return active1, active0, g1, g0, g2

# 自定义三元组损失函数
def triplet_loss(anchor, positive, negative, margin=1.0):
    # 标准化嵌入向量
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)

    # 计算距离
    pos_dist = torch.pairwise_distance(anchor, positive)
    neg_dist = torch.pairwise_distance(anchor, negative)

    # 计算三元组损失
    losses = torch.relu(pos_dist - neg_dist + margin)
    return losses.mean()

# 绘制并保存训练进度图表的函数
def plot_training_progress(contrast_losses, class_losses, accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(unsupervised_epochs), contrast_losses, label="first stage loss", color="red", linestyle="--",
             marker="^")
    plt.title("first stage loss")
    plt.xlabel("epoch")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig("第一阶段损失.jpg", dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(range(classifier_epochs), class_losses, label="loss of Classifier", color="blue", marker="o")
    plt.plot(range(classifier_epochs), accuracies, label="Classifier accuracy", color="green", marker="s")
    plt.title("Second stage loss")
    plt.xlabel("epoch")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.savefig("第二阶段损失.jpg", dpi=300)
    plt.show()
    print('图像完成')

# 用于在每个epoch后评估模型的函数
def evaluate_model(encoder, classifier, test_batch_idx=0):
    # 加载测试批次
    _, _, g1_test, g0_test, _ = listmake(test_batch_idx)
    test_graphs = g1_test + g0_test

    if len(test_graphs) == 0:
        return 0.0  # 如果没有测试数据，返回0准确率

    test_batch = Batch.from_data_list(test_graphs).to(device)

    # 创建标签
    test_labels = torch.zeros(len(test_graphs), dtype=torch.long, device=device)
    test_labels[:len(g1_test)] = 1

    # 确保模型处于评估模式
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        # 前向传递
        h = encoder(test_batch.x, test_batch.edge_index)
        g = global_mean_pool(h, test_batch.batch)
        out = classifier(g)

        # 计算准确率
        _, predicted = torch.max(out.data, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels) * 100  # 转换为百分比

    # 切换回训练模式
    encoder.train()
    classifier.train()

    return accuracy

# def test_model(A0,A1):
#     # 加载测试批次
#     for i in range(40):
#         A0 = random.sample(A0, 800)
#         A1 = random.sample(A1, 38)
#
#         g1_test = []
#         g0_test = []
#         for i in A1:
#             g1_test.append(int(extract_digits(i)))
#         print(g1_test)
#         for i in A0:
#             g0_test.append(int(extract_digits(i)))
#         print(g0_test)
#         g1_test = similarity.precompute_all_graphs(g1_test)
#         g0_test = similarity.precompute_all_graphs(g0_test)
#         test_graphs = g1_test + g0_test
#         print('正在进行最终模型评估1')
#         if len(test_graphs) == 0:
#             print("警告：测试批次为空！")
#             return 0, 0, 0, 0
#
#         test_batch = Batch.from_data_list(test_graphs).to(device)
#         print('正在进行最终模型评估2')
#         # 创建标签
#         test_labels = torch.zeros(len(test_graphs), dtype=torch.long, device=device)
#         test_labels[:len(g1_test)] = 1
#         accuracy1=precision1=recall1=f2=0
#         # 确保模型处于评估模式
#         encoder.eval()
#         classifier.eval()
#         print('正在进行最终模型评估3')
#         with torch.no_grad():
#             # 前向传递
#             h = encoder(test_batch.x, test_batch.edge_index)
#             g = global_mean_pool(h, test_batch.batch)
#             out = classifier(g)
#
#             # 计算准确率
#             _, predicted = torch.max(out.data, 1)
#             accuracy = (predicted == test_labels).sum().item() / len(test_labels)
#
#             # 计算每类的精确率和召回率
#             true_positive = ((predicted == 1) & (test_labels == 1)).sum().item()
#             false_positive = ((predicted == 1) & (test_labels == 0)).sum().item()
#             false_negative = ((predicted == 0) & (test_labels == 1)).sum().item()
#
#             precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
#             recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
#             f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#             accuracy1 +=accuracy
#             precision1+=precision
#             recall1+=recall
#             f2 +=f1
#         accuracy1 = accuracy1/40
#         precision1 = precision1/40
#         recall1 = recall1/40
#         f2 = f2/40
#     print(f"测试结果 - 准确率: {accuracy1:.4f}, 精确率: {precision1:.4f}, 召回率: {recall1:.4f}, F1分数: {f2:.4f}")
#     # return accuracy, precision, recall, f1

# print(__name__)

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    active_values, active_0_name, active_1_name = number.extract_active_property("A.sdf")
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备是{device}')
    # device = torch.device('cpu')#强制cpu运算
    matrix = matrix_make.matrix()
    # 超参数（根据需要定制）
    num_batches = len(active_1_name)  # listmake提供的批次总数
    batch_size = 400  # 每批图数（100个正图，100个负图）
    unsupervised_epochs = 200  # 对比（Triplet）训练的epoch数
    lr_contrast = 1e-3  # 对比学习优化器的学习率
    margin = 1.0  # 三元组损失的间隔参数

    classifier_epochs = 50  # 分类器训练的epoch数
    lr_classifier = 1e-3  # 分类器优化器学习率

    # 性能指标跟踪变量
    contrast_losses = []  # 对比损失历史
    class_losses = []  # 分类损失历史
    accuracies = []  # 准确率历史
    epochs = []  # 轮次标识

    
    # 获取一批样例来推断维度（每个节点的特征）
    try:
        _, _, g1_sample, g0_sample, g2_sample = listmake(50)  # 修改：正确解析listmake的五个返回值
        sample_graphs = g1_sample + g0_sample + g2_sample  # 合并正负样本图和锚点图
    except NameError:
        raise NameError("函数listmake(i)未定义。请确保提供了listmake函数。")

    if len(sample_graphs) == 0:
        raise ValueError("从listmake获取的批次为空。")
    num_node_features = sample_graphs[0].x.size(1)
    # 使用提供的GNN类定义GNN编码器
    hidden_dim1 = 64
    hidden_dim2 = 64
    output_dim = 32  # 图的嵌入维数
    gnn_input = {
        "LR": lr_contrast,
        "NAME": "Triplet_Encoder",
        "CHECKPOINT_DIR": "",
        "SIZE_LAYERS": [num_node_features, hidden_dim1, hidden_dim2, output_dim]
    }
    encoder = GNN(gnn_input).to(device)

    # 对比学习优化器（只包含编码器）
    optimizer_contrast = optim.Adam(encoder.parameters(), lr=lr_contrast)

    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_contrast, mode='min', factor=0.5, patience=10,
                                                           verbose=True)

    # Triplet（对比）训练循环
    print("开始对比学习训练(Triplet)...")
    encoder.train()
    for epoch in range(unsupervised_epochs):
        print(f'epoch:{epoch}')
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx in range(num_batches):
            print(f'第{batch_idx}')
            # 加载一批图（锚点、正样本和负样本）
            _, _, g1, g0, g2 = listmake(batch_idx)
            # input()
            # 跳过没有足够三元组样本的批次
            if not g2 or not g1 or not g0:
                continue

            # 确保每种类型至少有一个样本
            min_samples = min(len(g2), len(g1), len(g0))
            if min_samples == 0:
                continue

            # 统一样本数量以形成有效的三元组（取最小值以确保平衡）
            # g2 = g2[:min_samples]  # 锚点
            # g1 = g1[:min_samples]  # 正样本
            # g0 = g0[:min_samples]  # 负样本

            # 创建批次图对象
            batch_anchor = Batch.from_data_list(g2).to(device)
            batch_positive = Batch.from_data_list(g1).to(device)
            batch_negative = Batch.from_data_list(g0).to(device)

            # 编码节点嵌入
            h_anchor = encoder(batch_anchor.x, batch_anchor.edge_index)
            h_positive = encoder(batch_positive.x, batch_positive.edge_index)
            h_negative = encoder(batch_negative.x, batch_negative.edge_index)

            # 池化以获取图级嵌入
            z_anchor = global_mean_pool(h_anchor, batch_anchor.batch)
            z_positive = global_mean_pool(h_positive, batch_positive.batch)
            z_negative = global_mean_pool(h_negative, batch_negative.batch)
            # print(len(z_anchor))
            # print(len(z_positive))
            # print(len(z_negative))

            # 计算三元组损失
            loss = triplet_loss(z_anchor, z_positive, z_negative, margin)
            optimizer_contrast.zero_grad()
            loss.backward()
            optimizer_contrast.step()

            epoch_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            avg_epoch_loss = epoch_loss / batch_count
            contrast_losses.append(avg_epoch_loss)
            print(f"Epoch [{epoch + 1}/{unsupervised_epochs}], 三元组损失: {avg_epoch_loss:.4f}")
            print(f'所有损失{contrast_losses}')
            scheduler.step(avg_epoch_loss)
        else:
            print(f"Epoch [{epoch + 1}/{unsupervised_epochs}], 没有有效的三元组批次")
            contrast_losses.append(0)

    print("对比学习训练完成！")

    # 图级二分类的分类头
    classifier = nn.Sequential(
        nn.Linear(output_dim, (int(output_dim/2))),
        nn.ReLU(),
        nn.Linear((int(output_dim / 2)), (int(output_dim / 4))),
        nn.ReLU(),
        nn.Linear((int(output_dim/4)), 2)
    ).to(device)
    # classifier = nn.Linear(output_dim, 2).to(device)
    optimizer_classif = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr_classifier)
    criterion = nn.CrossEntropyLoss()

    print("开始分类器训练...")
    encoder.train()
    classifier.train()
    for epoch in range(classifier_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx in range(num_batches):
            # 加载用于分类的图批次
            print(f'第{batch_idx}')
            _, _, g1, g0, _ = listmake(batch_idx)  # 修改：正确解析listmake的五个返回值
            graphs = g1 + g0  # 合并正负样本图

            if len(graphs) == 0:
                continue

            current_batch_size = len(graphs)
            batch = Batch.from_data_list(graphs).to(device)

            # 创建标签：g1为正样本(1)，g0为负样本(0)
            labels = torch.zeros(current_batch_size, dtype=torch.long, device=device)
            labels[:len(g1)] = 1  # 修改：只将g1部分标记为1

            # 前向传递：编码和池化
            h = encoder(batch.x, batch.edge_index)
            g = global_mean_pool(h, batch.batch)  # 图级嵌入
            out = classifier(g)

            # 计算分类损失
            loss = criterion(out, labels)
            optimizer_classif.zero_grad()
            loss.backward()
            optimizer_classif.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 计算准确率
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算并存储这个epoch的平均损失和准确率
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        accuracy = 100 * correct / total if total > 0 else 0

        # 存储数据以便绘图
        class_losses.append(avg_loss)
        accuracies.append(accuracy)
        epochs.append(epoch + 1)

        print(f"Epoch [{epoch + 1}/{classifier_epochs}], 分类损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")

        # 每个epoch后在测试集上评估一次
        test_accuracy = evaluate_model(encoder, classifier)
        print(f"测试集准确率: {test_accuracy:.2f}%")

    print("分类器训练完成！")

    # 保存训练好的编码器和分类头
    if os.path.exists('encoder_classifier.pth'):
        os.remove('encoder_classifier.pth')
        print(f"encoder_classifier.pth已删除")
    else:
        print(f"encoder_classifier.pth不存在")
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'classifier_state_dict': classifier.state_dict()
    }, 'encoder_classifier.pth')
    print("模型已保存到encoder_classifier.pth")

    # 绘制训练进度
    if os.path.exists('第一阶段损失.jpg'):
        os.remove('第一阶段损失.jpg')
        print(f"第一阶段损失.jpg已删除")
    else:
        print(f"第一阶段损失.jpg不存在")
    if os.path.exists('第二阶段损失.jpg'):
        os.remove('第二阶段损失.jpg')
        print(f"第二阶段损失.jpg已删除")
    else:
        print(f"第二阶段损失.jpg不存在")
    plot_training_progress(contrast_losses, class_losses, accuracies)

    # 添加一个更详细的测试函数

    # 在保存模型后进行测试评估
    # print("正在进行最终模型评估...")
    # test_model(active_0_name , active_1_name)
