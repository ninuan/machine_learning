# 导入手写数字数据集
from sklearn.datasets import load_digits
import numpy as np

# 导入kmeans
from sklearn.cluster import KMeans
import kmeans1
# 导入metrics,计算accuracy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
digits = load_digits()

# # Sklearn的kmeans分类
# kmeans = KMeans(n_clusters=10,random_state=0)
# kmeans.fit(digits.data)
# clusters = kmeans.predict(digits.data)

# 自定义kmeans分类
kmeans = kmeans1.kmeans(data=digits.data,k=10)
clusters =kmeans.compute()

# 打印Clusters的中心点
fig,ax = plt.subplots(2,5,figsize=(8,3))
centers = kmeans.cluster_centers_.reshape(10,8,8)
fig.suptitle('Clusters visualization')
for axi,center in zip(ax.flat,centers):
    axi.set(xticks=[],yticks=[])

    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)


# 进行众数匹配
from scipy.stats import mode
labels = np.zeros_like(clusters)
print(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(digits.target[mask])[0] # [True,False]作为mask，选择数组元素;求众数；标量赋值给数组，自动广播
print(mask.shape)
print(labels.shape)

# 计算准确率,4位小数格式化输出
print(f'Sklearn的kmeans分类准确率为：{accuracy_score(digits.target,labels):.4f}')
plt.show()
