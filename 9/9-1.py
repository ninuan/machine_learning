# 1.使用Kmeans算法对给定2为数据集进行聚类分析（k=4），要求：
# （1）根据算法流程，手动实现Kmeans算法；
# （2）调用sklearn中聚类算法，对给定数据集进行聚类分析；
# （3）对比上述2中Kmeans算法的聚类效果。
# 2.调用sklearn中手写字体数据集（每个样本64维），使用kmeans进行聚类，将聚类结果作为分类结果，对比真实标签，计算分类精度（可以手动查看聚类标签）。

import numpy as np
import matplotlib.pyplot as plt

# 1.1 手动实现Kmeans算法
# 1.1.1 读取数据
data = np.loadtxt('data.txt')

# 随机选取k个样本作为初始聚类中心
def initCentroids(data, k):
    numSamples, dim = data.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = data[index, :]
    return centroids

# 计算两个向量的欧式距离
def euclDistance(vector1, vector2):
    return np.sqrt(sum(np.power(vector2 - vector1, 2)))

# 更新聚类中心的位置
def getCentroids(data, k, clusterAssment):
    numSamples, dim = data.shape
    centroids = np.zeros((k, dim))
    # 用于存储每个聚类中心下的样本数
    # clusterAssment[i,0]存储的是第i个样本所属的聚类中心
    # clusterAssment[i,1]存储的是第i个样本与其聚类中心的距离
    clusterCount = np.zeros((k, 1))
    for i in range(numSamples):
        # 获取第i个样本所属的聚类中心
        index = int(clusterAssment[i, 0])
        # 第i个样本与其聚类中心的距离
        centroids[index, :] += data[i, :]
        clusterCount[index, 0] += 1
    # 更新聚类中心的位置
    for i in range(k):
        centroids[i, :] /= clusterCount[i, 0]
    return centroids

# k为聚类中心的个数
def kmeans(data, k):
    numSamples = data.shape[0]
    # clusterAssment[i,0]存储的是第i个样本所属的聚类中心
    # clusterAssment[i,1]存储的是第i个样本与其聚类中心的距离
    clusterAssment = np.zeros((numSamples, 2))
    clusterChanged = True
    # 初始化聚类中心
    centroids = initCentroids(data, k)
    while clusterChanged:
        clusterChanged = False
        # 遍历每个样本
        for i in range(numSamples):
            # 最小距离
            minDist = 100000.0
            # 最近的聚类中心的索引
            minIndex = 0
            # 计算第i个样本与每个聚类中心的距离
            for j in range(k):
                distance = euclDistance(centroids[j, :], data[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 如果第i个样本所属的聚类中心发生了变化
            # 则clusterChanged为True
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                # 更新第i个样本所属的聚类中心及距离
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 更新聚类中心的位置
        centroids = getCentroids(data, k, clusterAssment)
    return centroids, clusterAssment

# 可视化聚类结果
def showCluster(data, k, centroids, clusterAssment):
    numSamples, dim = data.shape
    if dim != 2:
        print("数据不是二维的")
        return 1
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 绘制所有的样本
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(data[i, 0], data[i, 1], mark[markIndex])
    # 绘制聚类中心
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()

# 聚类中心的个数
k = 4
# 调用手动实现的Kmeans算法
centroids, clusterAssment = kmeans(data, k)
# 可视化聚类结果
showCluster(data, k, centroids, clusterAssment)
