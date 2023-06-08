# 实现k-means算法
# 1. 随机选择k个点作为初始聚类中心， k=2
# 2. 计算每个点到聚类中心的距离，将每个点分配到最近的聚类中心
# 3. 计算每个聚类的平均值，将该聚类的中心移动到平均值的位置
# 4. 重复2、3步骤，直到聚类中心不再变化

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from dataclasses import dataclass,field

@dataclass
class Point:
    features:np.ndarray
    cluster:int = 0


@dataclass
class Cluster:
    features:np.ndarray
    points:list[Point] = field(default_factory=list)

class kmeans:
    """
    Return:
        ret:更新后的最新Cluster_list
    """
    def __init__(self,data:np.ndarray,k=2):
        # data已经flatten过
        self.data = data
        self.k = k
        self.Cluster_list = []
        self.Point_list = []

        # 随机选择k个点作为初始聚类中心，创建 Cluster,加入list中管理
        random_index = np.random.choice(len(data),k,replace=False)
        for idx in random_index:
            cluster = Cluster(np.array(data[idx]))
            self.Cluster_list.append(cluster)
        # print(f'初始Cluster:\n {self.Cluster_list}')
        
        # 将各个点加入Point_list中管理
        for ele in data:
            ele =  np.array(ele)
            point = Point(np.array(ele))
            self.Point_list.append(point)
        # print(self.Point_list)
    # 一次 kmeans计算过程
    def forward(self):
        # 2.计算每个点到聚类中心的距离，将每个点分配到最近的聚类中心
        for point in self.Point_list:
            distance = []
            for cluster in self.Cluster_list:
                distance.append(np.linalg.norm(point.features-cluster.features))
            # 将该点分配到最近的聚类中心,并修改 点对应的cluster属性
            cluster.points.append(point)
            point.cluster = np.argmin(distance)
        # 划分后的Cluster包含的点
        # for cluster in self.Cluster_list:
            # print(f'cluster:{cluster}包含的点：{cluster.points}')

        # 3.计算每个聚类的平均值，将该聚类的中心移动到平均值的位置，并清空聚类的points
        for cluster in self.Cluster_list:
            if len(cluster.points) > 0:
                cluster.features = np.mean([point.features for point in cluster.points],axis=0)
                cluster.points = []
        # print(f'更新后的最新Cluster:\n {self.Cluster_list}')
        return self.Cluster_list
    def compute(self):
        ret = [] # 每个样本对应的cluster
        idx =0
        while True:
            self.forward()
            idx+=1
            if len(ret) > 1 and ret[-1] == ret[-2] or idx==50: #重复2、3步骤，直到聚类中心不再变化
                ret = self.forward()
                self.cluster_centers_ = np.array([cluster.features for cluster in ret])
                print(self.cluster_centers_.shape)
                ret = np.array([point.cluster for point in self.Point_list])
                return ret

            
                

    

