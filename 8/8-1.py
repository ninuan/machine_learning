# 实现PCA算法实现手写字体识别，要求：
# 1、实现手写数字数据集的降维；
# 2、比较两个模型（64维和10维）的准确率；
# 3、对两个模型分别进行10次10折交叉验证，绘制评分对比曲线。

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 加载手写数字数据集
digits = load_digits()
X = digits.data
y = digits.target

# 定义PCA模型和逻辑回归模型
pca64 = PCA(n_components=64)
pca10 = PCA(n_components=10)
logreg = LogisticRegression(solver='liblinear', max_iter=100)

# 定义Pipeline，将PCA和逻辑回归模型串联起来
model64 = Pipeline([('scaler', StandardScaler()), ('pca', pca64), ('logreg', logreg)])
model10 = Pipeline([('scaler', StandardScaler()), ('pca', pca10), ('logreg', logreg)])

# 1. 手写数字数据集的降维
X64 = pca64.fit_transform(X)
X10 = pca10.fit_transform(X)

# 2. 比较两个模型的准确率
accuracy64 = np.mean(cross_val_score(model64, X, y, cv=10))
accuracy10 = np.mean(cross_val_score(model10, X, y, cv=10))
print("64维模型准确率：", accuracy64)
print("10维模型准确率：", accuracy10)

# 3. 对两个模型进行10次10折交叉验证，并绘制评分对比曲线
kfold = StratifiedKFold(n_splits=10)

scores64 = []
scores10 = []

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model64.fit(X_train, y_train)
    score64 = model64.score(X_test, y_test)
    scores64.append(score64)

    model10.fit(X_train, y_train)
    score10 = model10.score(X_test, y_test)
    scores10.append(score10)

# 绘制评分对比曲线
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), scores64, label='64维模型')
plt.plot(range(1, 11), scores10, label='10维模型')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-validation Scores')
plt.legend()
plt.show()




