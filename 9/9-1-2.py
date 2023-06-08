# 调用sklearn中kmeans算法，对给定数据集进行聚类分析

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.loadtxt('data.txt')
# 调用kmeans算法，进行聚类分析
kmeans = KMeans(n_clusters=4)
kmeans.fit(data)
# 获取聚类中心
centroids = kmeans.cluster_centers_
# 获取聚类标签
labels = kmeans.labels_
# 绘制聚类结果
plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, c='r')
plt.show()

