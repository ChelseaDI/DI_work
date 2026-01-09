import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


class UserClustering:
    """
    在【单个子图 Loader】中执行用户聚类，并构建 cluster-level 表征。

    功能包括：
    1. 基于训练完成的 LightGCN 用户 embedding 做 KMeans
    2. 构建 cluster -> users 映射
    3. 计算每个 cluster 的虚拟用户 embedding
    4. 汇总每个 cluster 对应的物品集合（用于后续交互）
    """

    def __init__(self, n_clusters, seed=42):
        """
        参数：
            n_clusters : int
                子图内要划分的用户簇数量
            seed : int
                随机种子（保证可复现）
        """
        self.n_clusters = n_clusters
        self.seed = seed

    @torch.no_grad()
    def run(self, sub_loader, rec_model):
        """
        对一个子图执行用户聚类

        参数：
            sub_loader : Loader
                splitter.py 中构造的子图 Loader
                - sub_loader.n_user
                - sub_loader._allPos
                - sub_loader.group_id
            rec_model : LightGCN
                已在该子图上训练完成的模型

        返回：
            cluster_data : dict
                包含 cluster-level 的全部信息
        """

        # =========================================================
        # 1️⃣ 取子图内所有用户的 embedding
        # =========================================================
        user_emb = rec_model.embedding_user.weight.detach().cpu().numpy()       # (n_user, embedding_dim)
        n_user, emb_dim = user_emb.shape

        # =========================================================
        # 2️⃣ 使用 KMeans 对用户 embedding 聚类
        # =========================================================
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(user_emb)       # cluster_labels[u] = 用户 u 所属的 cluster id

        # =========================================================
        # 3️⃣ 构建 cluster -> users 映射
        # =========================================================
        cluster_users = defaultdict(list)
        for local_uid, cid in enumerate(cluster_labels):
            cluster_users[cid].append(local_uid)

        # =========================================================
        # 4️⃣ 计算 cluster embedding（虚拟用户节点）
        # =========================================================
        # 使用簇内用户 embedding 的均值
        cluster_embeddings = {}
        for cid, uids in cluster_users.items():
            cluster_embeddings[cid] = user_emb[uids].mean(axis=0)
            # shape: (embedding_dim,)

        # =========================================================
        # 5️⃣ 构建 cluster -> item 集合
        # =========================================================
        # 一个 cluster 覆盖的物品 = 簇内所有用户交互过的物品并集
        cluster_items = {}
        for cid, uids in cluster_users.items():
            items = set()
            for u in uids:
                items.update(sub_loader._allPos[u])         # update 是集合的批量加入操作
            cluster_items[cid] = np.array(list(items), dtype=np.int64)

        # =========================================================
        # 6️⃣ 整理输出（标准化结构，方便后续使用）
        # =========================================================
        cluster_data = {
            "group_id": sub_loader.group_id,          # 子图编号
            "n_clusters": self.n_clusters,            # cluster 数量
            "cluster_users": dict(cluster_users),     # cid -> [local_uid]
            "cluster_embeddings": cluster_embeddings, # cid -> embedding
            "cluster_items": cluster_items            # cid -> item ids
        }

        return cluster_data
