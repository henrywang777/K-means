# _*_ coding : utf-8
# @Time : 2022/11/18 15:48
# @Author : 施树含
# @File ：DS_LPA
# @Project : 新建文件夹

import networkx as nx
import matplotlib.pyplot as plt
from numpy import *
import math
import pandas as pd
from sklearn.cluster import KMeans


# 绘制初始数据图
class Graph:
    # 这一步可以省略
    graph = nx.Graph()

    def __init__(self):
        self.graph = nx.Graph()

    # 根据txt文件创建图
    def createGraph(self, filename):
        file = open(filename, 'r')
        for line in file.readlines():
            nodes = line.split()
            edge = (int(nodes[0]), int(nodes[1]))
            self.graph.add_edge(*edge)

        return self.graph


# 节点类 存储社区与节点编号信息
class Vertex:
    # 定义实例变量
    def __init__(self, vid, cid):
        # 节点编号
        self._vid = vid
        # 社区编号
        self._cid = cid

        # 节点的度-----deg

        # 节点的局部密度------density

        # 节点的最小距离-----distance

        # 节点的星密度(density*)-----density_star

        # 节点的星距离(distance*)----distance_star

        # 节点的v*-------------------v_star
        self.v_star = 0


class Ds_Lpa():
    # 用来设定epsilion---目前自己设定的固定值
    epsilion = 1.53
    # 用来记录节点最大局部密度
    nodes_density_max = 0
    # 用来记录节点最小局部密度
    nodes_density_min = 0
    # 用来记录jaccard指数---二维列表实现----实际存的是2点间距离的值
    jaccard_list = list()
    # 用来记录节点的最大距离
    nodes_distance_max = 0
    # 用来记录节点的最小距离
    nodes_distance_min = 0
    # 用来记录所有节点的v_star -----(vi)----列表
    vi_list = list()
    # 用来记录所有节点的期望值
    e = 0
    # 用来记录所有节点的标准差
    sd = 0
    # 用来记录区分中心节点和普通节点的边界值
    bound = 0
    # 记录社区数量
    cid = 1
    # 存储中心节点--->列表
    center = list()

    # 计算每个节点的度
    def cal_degree(self, G, allnodeslist, final_nodes):
        k = 0
        for i in allnodeslist:
            final_nodes[k].deg = G.degree(i)
            k += 1

    # 计算每个节点的局部密度
    def cal_local_density(self, G, allnodeslist, final_nodes):
        k = 0
        for x in allnodeslist:
            # 记录某个节点的局部密度----node_density
            # 记录某个节点的邻居节点的度之和
            neighbor_nodes_degnum = 0
            # 记录某个节点的领域密度 ----neighbor_density
            for neighbor in list(G.neighbors(x)):
                j = allnodeslist.index(neighbor)
                neighbor_nodes_degnum += final_nodes[j].deg

            neighbor_density = G.degree(x) + neighbor_nodes_degnum
            # node_density = G.degree(x) + (G.degree(x) / neighbor_density) * neighbor_nodes_degnum
            node_density = G.degree(x) + (1 / neighbor_density) * neighbor_nodes_degnum

            if node_density > self.nodes_density_max:
                self.nodes_density_max = node_density
            # 把x节点的局部密度存储在节点实例中
            final_nodes[k].density = node_density
            k += 1

    # 计算每个节点到不同节点的距离
    def cal_distance(self, G, allnodeslist):
        # 计算Jaccard指数，二维列表
        for i in allnodeslist:
            sij_row = list()
            for j in allnodeslist:
                ni = list(G.neighbors(i))
                # i节点的邻居节点以及它自己的列表
                ni.append(i)
                # j节点的邻居节点以及它自己的列表
                nj = list(G.neighbors(j))
                nj.append(j)
                # 求2个集合的交集，返回给res列表
                res = list(set(ni) & set(nj))
                # 求2个集合的并集，返回给ress的列表
                ress = list(set(ni).union(nj))
                # try:
                sij = len(res) / len(ress)
                # sij = 2*(len(res)) / (len(ress)+len(res))

                # except ZeroDivisionError:
                #    sij = 1
                distance = 0
                if j == i:
                    distance = 0
                else:
                    distance = 1 / (sij + 1)
                    # distance = 1 - sij
                sij_row.append(distance)
            self.jaccard_list.append(sij_row)

    # 计算每个节点的最小距离
    def cal_distance_min(self, G, allnodeslist, final_nodes):
        k = 0
        for i in allnodeslist:
            if final_nodes[k].density == self.nodes_density_max:
                final_nodes[k].distance = max(self.jaccard_list[k])

                if final_nodes[k].distance > self.nodes_distance_max:
                    self.nodes_distance_max = final_nodes[k].distance

            else:
                # 临时存放密度大于i节点的节点序号列表
                temporary_index = list()
                # 临时存放所有大于i节点的密度值的节点的最小距离值列表
                temporary_value = list()
                temp = 0
                for j in allnodeslist:
                    if final_nodes[temp].density > final_nodes[k].density:
                        temporary_index.append(temp)
                    temp += 1
                for m in temporary_index:
                    temporary_value.append(self.jaccard_list[k][m])

                final_nodes[k].distance = min(temporary_value)

                if final_nodes[k].distance > self.nodes_distance_max:
                    self.nodes_distance_max = final_nodes[k].distance
            k += 1

    # 计算每个节点的边界值
    def cal_boundary_value(self, G, allnodeslist, final_nodes):
        # 从节点实例中选取最小距离和最小密度
        # 记录所有节点最小密度的列表
        all_density_list = list()
        # 记录所有节点的最小距离的列表
        all_distance_list = list()
        k = 0
        for i in allnodeslist:
            all_density_list.append(final_nodes[k].density)
            all_distance_list.append(final_nodes[k].distance)
            k += 1

        self.nodes_density_min = min(all_density_list)
        self.nodes_distance_min = min(all_distance_list)
        # 计算--密度*---最小距离*
        temp = 0
        for i in allnodeslist:
            final_nodes[temp].density_star = (final_nodes[temp].density - self.nodes_density_min) / (
                    self.nodes_density_max - self.nodes_density_min)
            final_nodes[temp].distance_star = (final_nodes[temp].distance - self.nodes_distance_min) / (
                    self.nodes_distance_max - self.nodes_distance_min)
            final_nodes[temp].v_star = final_nodes[temp].density_star * final_nodes[temp].distance_star
            self.vi_list.append(final_nodes[temp].v_star)
            temp += 1

    # 计算确切的边界值（切比雪夫不等式）and 输出中心节点
    def cal_center_nodes(self, allnodeslist, final_nodes):
        # 用于记录临时总和
        temp = 0
        # 计算期望值
        self.e = mean(self.vi_list)

        # 计算标准差
        for i in self.vi_list:
            temp = temp + (i - self.e) * (i - self.e)

        self.sd = math.sqrt(temp / (len(self.vi_list) - 1))
        self.bound = self.e + self.epsilion * self.sd
        # 输出中心节点
        k = 0
        for i in allnodeslist:
            if final_nodes[k].v_star > self.bound:
                self.center.append(i)
            k += 1

    def K_means_clustering(self, data, final_nodes, random_state):
        kmeans = KMeans(n_clusters=2, random_state=random_state)
        kmeans.fit(data)
        labels = kmeans.labels_
        H_group = []
        L_group = []
        for index, lable in enumerate(labels):
            if lable == 0: H_group.append(final_nodes[index])
            if lable == 1: L_group.append(final_nodes[index])
        return H_group, L_group
        # L_group实例访问不到v_star属性怎么办？？？？？？？后面的Vc就也访问不到
        # 在将 labels 的结果映射回 final_nodes 时出现了问题，index对应不上

    def Cluster_Center_Selection(self, allnodeslist, final_nodes, random_state):
        Vc = final_nodes
        while True:
            H_group, L_group = self.K_means_clustering(Vc, final_nodes, random_state)
            self.center = self.center.extend(H_group)
            Vc = L_group


if __name__ == '__main__':

    # 导入txt文件绘制成相应无向图
    G = Graph().createGraph('club.txt')
    # nx.draw(G)
    # plt.show()

    # 把图的所有节点保存到列表中,allnodes_list也充当参考顺序
    allnodeslist = list(G.nodes())
    # 打印所有节点信息
    print(allnodeslist)

    # final_list和allnodes_list一一对应，这样就可以通过allnodes_list里节点的顺序，找到final_list中
    # 相应的顺序
    # 创建对象列表存放节点实例
    final_nodes = list()
    for i in allnodeslist:
        final_nodes.append(Vertex(i, i))

    ex = Ds_Lpa()

    ex.cal_degree(G, allnodeslist, final_nodes)

    ex.cal_local_density(G, allnodeslist, final_nodes)

    ex.cal_distance(G, allnodeslist)

    ex.cal_distance_min(G, allnodeslist, final_nodes)

    ex.cal_boundary_value(G, allnodeslist, final_nodes)

    ex.cal_center_nodes(allnodeslist, final_nodes)

    # 输出中心节点列表
    print(ex.center)
