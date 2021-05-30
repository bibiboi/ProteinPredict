# ProteinPredict
利用PageRank（重启随机游走）预测蛋白质相互作用



</font>



<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 算法描述

<font color=#999AAA ></font>

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

<font color=#999AAA >重启随机游走算法（Random Walk with Restart）
重启随机游走算法是在随机游走算法的基础的改进。从图中的某一个节点出发，每一步面临两个选择，随机选择相邻节点，或者返回开始节点。算法包含一个参数a为重启概率，1-a表示移动到相邻节点的概率，经过迭代到达平稳，平稳后得到的概率分布可被看作是受开始节点影响的分布。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027030108226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)

# 设计思想
*要求实现：*

<font color=#999AAA >
  1. 获得相互作用数最多的蛋白质P
  
2.  预测与P相互作用的Top-20蛋白质
  
   3. P在五折交叉验证下的ROC
   
   *设计思想：*
1. 解析文本数据获取相互作用数最多的蛋白质P及相关作用关系矩阵
2. 初始化r0向量，进行迭代计算，迭代10次后（10次是因为PageRank迭代10次后基本稳定）分析r的值得到与P相互作用的Top-20蛋白质
3. 随机将与蛋白质P相互作用的蛋白质分成5个子集，依次将其相互作用抹去，进行计算r并画出ROC曲线

# 源代码及运行结果
## 运行结果


<font color=#999AAA >运行结果如下：
  
![在这里插入图片描述](https://img-blog.csdnimg.cn/20201027034224701.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpYmliaWJpYm9p,size_16,color_FFFFFF,t_70#pic_center)



## 源代码

<font color=#999AAA >代码如下：




```python
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
```

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">

# 总结

这次算法实现的难点主要在于理解，最初实现时尝试采用PageRank思想理解这个预测过程，发现虽然PageRank和重启随机游走公式相同，但思想还是不太一样，又因为查找到有关重启随机游走算法的资料非常少而且不太容易理解，这就导致最初对于如何实现毫无头绪。

但在理解整个算法后实现起来并没有太大的问题，只是因为数据集略大导致跑完所有代码需要一点时间，如何存储数据并且改进算法使其提高效率非常需要优化的一点。实现过程中对于如何随机选取测试样本进行五折交叉验证并且画出ROC曲线上花了点时间，最后想出了自认为比较满意且简单的方法。

众多不足，日后改进。
