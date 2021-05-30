#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import copy
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#计算PR值
def calPR(W,connectionNum,proteinNum,maxIndex):
    W = W / connectionNum #归一化
    r = np.zeros(proteinNum) # 初始化r向量
    r[maxIndex] = 1
    r = np.mat(r).transpose()
    r2 = copy.deepcopy(r)
    W = np.mat(W)
    a = 0.85
    for i in range(10):#设置迭代次数：10
        r1 = copy.deepcopy(r2)
        r2 = a * W * r1 + (1 - a) * r
    return r2

# n*1维矩阵转换为列表
def matrixTolist(matrix):
    list = matrix.transpose().tolist()
    list = [i for ind in list for i in ind]
    return list

filename = 'BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt'
allprotein = [] # 存储所有蛋白质
relation  = [] # 存储蛋白质结构相互作用关系
with open(filename, 'r') as f:
    while True:
        lines = f.readline()  # 整行读取数据
        if not lines:
            break
            pass
        allprotein.append(lines.split()[0])
        allprotein.append(lines.split()[3])# 存储蛋白质数据
        relation.append((lines.split()[0],lines.split()[3])) # 存储蛋白质结构相互作用关系
    allprotein = list(set(allprotein))
    relation = list(set(relation))
proteinNum = len(allprotein)
W = np.zeros((proteinNum,proteinNum)) # 构建蛋白质相互作用关系矩阵
for rel in relation:
    W[allprotein.index(rel[0]),allprotein.index(rel[1])] = 1
    W[allprotein.index(rel[1]), allprotein.index(rel[0])] = 1

W0 = copy.deepcopy(W)
connectionNum = W.sum(axis = 0) # 计算每个蛋白质的相互作用数
maxConnectionNum = np.max(W.sum(axis = 0))
index = np.argmax(W.sum(axis = 0)) # 获取相互作用最多的蛋白质P及作用数及索引
maxProtein = allprotein[index]
print("======================================")
print("相互作用最多的蛋白质为：",maxProtein)
proteinLabel = W[:,index]
r2 = calPR(W,connectionNum,proteinNum,index)
#与P有相互作用Top20蛋白质
maxTwentyIndex = matrixTolist(r2)
maxTwentyIndex = np.argsort(maxTwentyIndex)[-21:] # 排序获取前20蛋白质
maxTwentyIndex = maxTwentyIndex[::-1]
print("======================================")
print("与" + maxProtein + "相互作用Top20蛋白质：")
rank = 1
for ind in maxTwentyIndex:
    if allprotein[ind] != maxProtein:
        print(rank ,":",allprotein[ind])
        rank += 1

#五折交叉验证并画出ROC曲线
nonZeroIndex = np.nonzero(proteinLabel)[0].tolist() # 获取标签值不为0的索引 即正例索引值
testIndex = [] # 存储每次测试样本的索引
for j in range(5):
    eachTestIndex = [nonZeroIndex[ind] for ind in np.random.choice(len(nonZeroIndex),
                                int(maxConnectionNum/5),replace=False)] # 5次依次随机抽取正例测试样本
    testIndex.append(eachTestIndex)
    nonZeroIndex = list(set(nonZeroIndex) - set(eachTestIndex)) # 将本次随机抽取的样本删除以免测试样本重复
xFPR = [] # 存储每次交叉验证的FPR和TPR
yTPR = []
for eachTest in testIndex: # 分别5次计算FPR和TPR
    W1 = copy.deepcopy(W0)
    for eachInd in eachTest: # 重置测试样本的相互作用关系
        W1[eachInd,index] = 0
        W1[index,eachInd] = 0
    connectionNum2 = W.sum(axis=0)
    R = calPR(W1,connectionNum2,proteinNum,index)
    zeroIndex = np.where(proteinLabel == 0)[0].tolist() # 获取标签值为0的索引 即反例索引值
    allTestIndex = eachTest + zeroIndex # 所有测试集的索引值
    R = matrixTolist(R)
    R = [R[i] for i in range(len(R)) if (i in allTestIndex)] # 去除非测试集的数据
    testProtein = [allprotein[i] for i in range(len(allprotein)) if (i in allTestIndex)] # 测试集数据对应的蛋白质
    testProteinP = [allprotein[i] for i in range(len(allprotein)) if (i in eachTest)] # 正例蛋白质集
    testProteinN = [allprotein[i] for i in range(len(allprotein)) if (i in zeroIndex)] # 反例蛋白质集
    proteinR = dict(zip(testProtein,R)) # {蛋白质 - r} 键值对
    orderProteinR = sorted(proteinR.items(),key = lambda x : x[1] ,reverse=True) # 进行排序
    orderProtein = [p[0] for p in orderProteinR] # 排序后的蛋白质
    x = [] # 每条ROC曲线的x值
    y = [] # 每条ROC曲线的y值
    for i in range(len(orderProtein)): # 进行计算 ： 依次将第i个值作为阈值
        predictP = orderProtein[:i+1] # 预测正例为前i个
        predictN = orderProtein[i+1:] # 预测反例为第i+1个到最后一个
        TP = len(set(predictP) & set(testProteinP)) # 真正例数
        TPR = TP/len(testProteinP)
        FPR = 1 - len(set(predictN) & set(testProteinN))/len(testProteinN) # 假正例率 = 1 - 真反例率
        x.append(FPR),y.append(TPR)
    xFPR.append(x),yTPR.append(y)

for i in range(5):
    label =  "第" + str(i + 1) + "次测试"
    plt.plot(xFPR[i],yTPR[i],alpha = 0.8,label = label)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()