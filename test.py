import re
import similarity
import matrix_make
import os
import check

# Example usage

def listmake1(a):
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


    # print('\n',active1,'\n',active0,'\n',a)
    g1, _, _ = similarity.precompute_all_graphs(active1)
    g0, _, _ = similarity.precompute_all_graphs(active0)
    g2, _, _ = similarity.precompute_all_graphs([a])



    return active1, active0, g1, g0, g2

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Replace these with your actual test lists

    matrix = matrix_make.matrix()
    # Path to saved model
    model_path = 'encoder_classifier.pth'
    # 获取一批样例来推断维度（每个节点的特征）
    try:
        _, _, g1_sample, g0_sample, g2_sample = listmake1(7)  # 修改：正确解析listmake的五个返回值
        sample_graphs = g1_sample + g0_sample + g2_sample  # 合并正负样本图和锚点图
    except NameError:
        raise NameError("函数listmake(i)未定义。请确保提供了listmake函数。")
    if len(sample_graphs) == 0:
        raise ValueError("从listmake获取的批次为空。")
    num_node_features = sample_graphs[3].x.size(1)
    print(sample_graphs[0])
    print(num_node_features)
    # input()
    # Number of features per node (should match training)
    # num_node_features = 0  # Replace with actual node feature dimension
    print('test_model开始')
    # Run the test
    check.test_model(model_path, num_node_features)