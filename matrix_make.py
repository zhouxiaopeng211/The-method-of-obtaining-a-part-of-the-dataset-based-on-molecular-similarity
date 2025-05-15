import csv
import os
from os import close
from rdkit import Chem
import numpy as np
import pandas as pd
import number
import heapq
def different_active_matrix(active_values, active_0_name, active_1_name ,A):
    #不同的性质（相似度1号）
    matrix = A
    number0 = []
    name0 = []
    number1 = []
    name1 = []
    similarity = []

    different_active = pd.read_csv('nr-ahr_similar(different_active).csv')
    number1.extend(different_active['Molecular number1'].tolist())
    number0.extend(different_active['Molecular number2'].tolist())
    name1.extend(different_active['DSSTox_CID_name1'].tolist())
    name0.extend(different_active['DSSTox_CID_name2'].tolist())
    similarity.extend(different_active['similarity'].tolist())
    # print(number1[0:100])
    # print(number0[0:100])
    # print(name1[0:100])
    # print(name0[0:100])
    # input()

    for a0 in range(0,len(active_1_name)):#380
    # for a0 in range(0, 2):  # test
    #     print(f'a0为',a0)
        data = similarity[a0*len(active_0_name):(a0+1)*len(active_0_name)]#8965
        k = 200
        largest_elements = heapq.nlargest(k, enumerate(data), key=lambda x: x[1])
        values = [x[1] for x in largest_elements]
        indices = [x[0] for x in largest_elements]

        # print(f"最大的{len(values)}个数值为:", values)
        # print(f"对应的位置索引为:", indices)
        # input()
        name = []
        number = []
        for a in indices:
            number.append(number0[a])
            name.append(name0[a])
        # print(f'对应的名字为', name)
        # print(f'对应的数字为', number)
        for a1 in range(0,k):
            # print(a0,a1)
            # print(name0[a0 * len(active_0_name) + a1])
            # print(a0 * len(active_0_name) + a1)
            matrix[a0, number[a1]] = '-'+name[a1]
            # print(matrix[a0, number[a1]])
    return matrix

def active1_matrix(active_values, active_0_name, active_1_name ,A):
    # 相同的性质（相似度2号）
    matrix = A
    number0 = []
    name0 = []
    number1 = []
    name1 = []
    similarity = []

    different_active = pd.read_csv('nr-ahr_similar(same1_active).csv')
    number1.extend(different_active['Molecular number1'].tolist())
    number0.extend(different_active['Molecular number2'].tolist())
    name1.extend(different_active['DSSTox_CID_name1'].tolist())
    name0.extend(different_active['DSSTox_CID_name2'].tolist())
    similarity.extend(different_active['similarity'].tolist())
    # print(len(number1))
    # print(len(number0))
    # print(len(name1))
    # print(len(name0))
    # input()

    for a0 in range(0, len(active_1_name)):  # 380
        # print(len(active_1_name))
        # input()
    # for a0 in range(0, 2):  # test
    #     print(f'a0为',a0)
        data = similarity[a0 * len(active_1_name):(a0 + 1) * len(active_1_name)]  # 380
        k = 200
        largest_elements = heapq.nsmallest(k, enumerate(data), key=lambda x: x[1])
        values = [x[1] for x in largest_elements]
        indices = [x[0] for x in largest_elements]

        # print(f"最大的{len(values)}个数值为:", values)
        # print(f"对应的位置索引为:", indices)
        name = []
        number = []
        for a in indices:
            number.append(number0[a])
            name.append(name0[a])
        # print(f'对应的名字为', name)
        # print(f'对应的数字为', number)
        # input()
        for a1 in range(0,k):
            # print(a1)
            # print(name0[a0 * len(active_0_name) + a1])
            # print(a0 * len(active_0_name) + a1)
            matrix[a0, number[a1]] = '+'+name[a1]
            # print(a0, number[a1])
            # print(matrix[a0, number[a1]])
        # print(matrix[1, 1741])
    return matrix




def matrix():
    active_values, active_0_name, active_1_name = number.extract_active_property("A.sdf")
    Matrix = np.zeros((len(active_1_name), len(active_0_name) + len(active_1_name)), dtype=list)  # 矩阵380*(8965+380)
    # Matrix = np.zeros((2,len(active_0_name) * len(active_1_name)), dtype=list)
    print(Matrix.shape)
    # if os.path.exists('matrix.csv'):
    #     os.remove('matrix.csv')
    #     print(f"文件  已删除")
    # else:
    #     print(f"文件  不存在")
##########################################################################################
#不同的性质（相似度1号）
    Matrix=different_active_matrix(active_values, active_0_name, active_1_name,Matrix)
##########################################################################################
#相同的性质（相似度2号）
    # print(Matrix.shape)
    # input()
    Matrix=active1_matrix(active_values, active_0_name, active_1_name,Matrix)
    # input()
    # print(Matrix[324,198])
    # np.save('matrix.npy', Matrix)
    # np.savetxt('matrix.csv', Matrix, fmt='%s', delimiter=',')
    return Matrix
# a=matrix()
# matrix_liat=a[1,:]
# print(a.shape)
# print(len(matrix_liat))
# print(a[364,1525])
# for i in range(0,len(matrix_liat)):
#     print(matrix_liat[i])