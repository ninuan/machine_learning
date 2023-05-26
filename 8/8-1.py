# 实现PCA算法实现手写字体识别，要求：
# 1、实现手写数字数据集的降维；
# 2、比较两个模型（64维和10维）的准确率；
# 3、对两个模型分别进行10次10折交叉验证，绘制评分对比曲线。

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

# 加载数据集
digits = datasets.load_digits()
X = digits.data
y = digits.target

# 降维
pca = PCA(n_components=10)
X_r = pca.fit(X).transform(X)

# 10折交叉验证
scores = cross_val_score(pca, X, y, cv=10)

