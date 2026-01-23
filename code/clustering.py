import torch
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from dataloader import BasicDataset


class UserClustering:
    """
    在【单个子图 Loader】中执行用户聚类，并构建 cluster-level 表征。

    功能包括：
    1. 基于训练完成的 LightGCN 用户 embedding 做 KMeans
    2. 构建 cluster -> users 映射
    3. 构建 cluster-level 虚拟用户 embedding（torch）
    4. 汇总每个 cluster 的【训练交互】物品集合
    """

    def __init__(self, n_clusters, seed=42):
        self.n_clusters = n_clusters
        self.seed = seed

    @torch.no_grad()
    def run(self, sub_loader:BasicDataset, rec_model):
        """
        对一个子图执行用户聚类

        参数：
            sub_loader : Loader
                子图 Loader（只依赖公共接口）
            rec_model : LightGCN
                已在该子图上训练完成的模型

        返回：
            cluster_data : dict
        """

        # =========================================================
        # 1️⃣ 获取子图用户 embedding（torch）
        # =========================================================
        user_emb: torch.Tensor = rec_model.embedding_user.weight.detach()
        n_user, emb_dim = user_emb.shape

        # =========================================================
        # 2️⃣ KMeans（sklearn 只能吃 numpy）
        # =========================================================
        user_emb_np = user_emb.cpu().numpy()        # (n_user, embedding_dim)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10
        )
        cluster_labels = kmeans.fit_predict(user_emb_np)        # cluster_labels[u] = 用户 u 所属的 cluster id

        # =========================================================
        # 3️⃣ cluster -> users
        # =========================================================
        cluster_users = defaultdict(list)
        for local_uid, cid in enumerate(cluster_labels):
            cluster_users[cid].append(local_uid)

        # =========================================================
        # 4️⃣ cluster embedding（torch mean）
        # =========================================================
        cluster_embeddings = {}
        for cid, uids in cluster_users.items():
            uids_tensor = torch.tensor(uids, dtype=torch.long, device=user_emb.device)
            cluster_embeddings[cid] = user_emb[uids_tensor].mean(dim=0)
            # shape: (embedding_dim,)

        # =========================================================
        # 5️⃣ cluster -> items（仅 train 正样本）
        # =========================================================
        cluster_items = {}
        for cid, uids in cluster_users.items():
            items = set()
            # 使用公共接口，避免依赖 _allPos
            pos_items = sub_loader.getUserPosItems(uids)    # list
            for item_list in pos_items:
                items.update(item_list)      # update 是集合的批量加入操作
            cluster_items[cid] = np.fromiter(items, dtype=np.int64)

        # =========================================================
        # 6️⃣ 标准化输出（server 端可直接用）
        # =========================================================
        cluster_data = {
            "group_id": sub_loader.group_id,
            "n_clusters": self.n_clusters,
            "cluster_users": dict(cluster_users),          # cid -> [local_uid]
            "cluster_embeddings": cluster_embeddings,      # cid -> torch.Tensor
            "cluster_items": cluster_items                 # cid -> np.ndarray (train only)
        }

        return cluster_data
