import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
import os

import dataloader
import model
import utils
import Procedure
from world import cprint

class ServerGraph:
    """
    Server-level graph convolution over cluster virtual nodes.

    功能：
    1. 收集所有子图中的 cluster embedding
    2. 计算 cluster 间相似度
    3. 为每个 cluster 选 Top-K 相似 cluster
    4. 基于 cluster-item 覆盖关系进行一层卷积
    5. 输出更新后的 cluster embedding
    """

    def __init__(self, config: dict, n_items, device="cpu"):
        self.device = device
        self.n_items = n_items
        self.config = config

        self.n_clusters_all = self.config['n_clusters_all']
        self.latent_dim = self.config['latent_dim_rec']
        self.topk = self.config['server_topk']
        self.warm_start = self.config['server_warm_start']
        self.server_graph_enhance = self.config['server_graph_enhance']

        self.item_emb_buffer = []

    def reset(self):
        """
        清空上一轮 server 聚合过程中缓存的状态
        """
        self.item_emb_buffer = []

    # =========================================================
    # [MODIFIED] 解析 cluster 数据（torch-safe，不再 numpy stack）
    # =========================================================
    def _parse_clusters(self, cluster_data_list):
        # 用于将各 group 的 clusters data 平铺
        # [ [group1_cluster1], [group1_cluster2], [group2_cluster1], ... ]
        cluster_embs = []
        cluster_items = []
        cluster_keys = []       # 用于索引 cluster

        for data in cluster_data_list:    # 遍历每组的 clusters
            gid = data["group_id"]
            for cid, emb in data["cluster_embeddings"].items():
                cluster_keys.append((gid, cid))
                cluster_embs.append(emb)                          # emb 已是 torch.Tensor
                cluster_items.append(set(data["cluster_items"][cid]))

        # [MODIFIED] torch.cat + stack（避免 numpy → torch 往返）
        cluster_embs = torch.stack(cluster_embs, dim=0).to(self.device)     # (n_clusters_all, dim)

        return cluster_embs, cluster_items, cluster_keys

    # =========================================================
    # 相似度计算
    # =========================================================
    def _calc_cluster_sim(self, cluster_embs, cluster_items, mode="jaccard"):
        """
        mode:
            - "cosine"
            - "jaccard"
        """
        if mode == "cosine":
            return self._calc_cluster_sim_by_cosine(cluster_embs)
        if mode == "jaccard":
            return self._calc_cluster_sim_by_jaccard(cluster_items)
        raise ValueError(f"Unknown sim mode: {mode}")

    def _calc_cluster_sim_by_cosine(self, cluster_embs):
        """
        cluster_embs: Tensor, (n_cluster, dim)
        """
        # cosine similarity = normalized dot product
        norm_emb = F.normalize(cluster_embs, dim=1)            # 对张量按行归一化  a·b = ||a||·||b||·cos(θ) = cos(θ)
        sim_matrix = torch.matmul(norm_emb, norm_emb.t())      # (n_cluster, n_cluster)
        sim_matrix.fill_diagonal_(-1e9)                        # 自己对自己 -- 负无穷

        return sim_matrix

    def _calc_cluster_sim_by_jaccard(self, cluster_items):
        """
        cluster_items: List[Set[item_id]]
        """
        n_clusters_all = len(cluster_items)                             # [MODIFIED]
        try:
            assert n_clusters_all == self.n_clusters_all 
        except AssertionError:
            print(f"!!!!! n_clusters_all: {n_clusters_all}, self.n_clusters_all: {self.n_clusters_all}")
        
        sim_matrix = torch.zeros((n_clusters_all, n_clusters_all), device=self.device)
        for i in range(n_clusters_all):
            items_i = cluster_items[i]
            for j in range(i + 1, n_clusters_all):
                items_j = cluster_items[j]
                inter = len(items_i & items_j)
                if inter == 0:
                    continue
                union = len(items_i | items_j)
                sim = inter / union
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        sim_matrix.fill_diagonal_(-1e9)
        return sim_matrix

    # =========================================================
    # 子图上传 item embedding
    # =========================================================
    def collect_item_embeddings(self, item_emb):
        """
        item_emb : Tensor, shape = (n_item, dim)
                来自某一个子图，必须是 computer() 后的结果
        """
        self.item_emb_buffer.append(item_emb.detach().cpu())    # 统一放在 CPU，节省显存

    # 聚合 item embedding（均值）
    def aggregate_item_embeddings(self):
        """
        return:
            server_item_emb : Tensor, shape = (n_item, dim)
        """
        assert len(self.item_emb_buffer) > 0, "No item embeddings collected!"
        stacked = torch.stack(self.item_emb_buffer, dim=0)      # (n_group, n_item, dim)
        server_item_emb = stacked.mean(dim=0)                   # (n_item, dim)
        return server_item_emb.to(self.device)

    # =========================================================
    # Top-K 邻居选择
    # =========================================================
    def _select_topk_neighbors(self, sim_matrix):
        """
        Select Top-K similar clusters for each cluster.
        """
        # 取 Top-K 相似 cluster，构建 server-level 的 User–Item 图
        _, topk_idx = torch.topk(sim_matrix, self.topk, dim=1)      # topk_idx: (n_cluster, topk)
        return topk_idx

    # =========================================================
    # 构图逻辑（cluster → item）
    # =========================================================
    def _build_server_graph(self, sim_matrix, cluster_items, mode="self"):
        if mode == "self":
            UserItemNet, train_user_all, train_item_all = self._build_graph_self_items(cluster_items)
            return UserItemNet, train_user_all, train_item_all
        if mode == "topk":
            UserItemNet, train_user_all, train_item_all = self._build_graph_topk_cluster_items(sim_matrix, cluster_items)
            return UserItemNet, train_user_all, train_item_all
        if mode == "self+topk":
            UserItemNet, train_user_all, train_item_all = self._build_graph_self_and_topk_items(sim_matrix, cluster_items)
            return UserItemNet, train_user_all, train_item_all
        raise ValueError(f"Unknown graph mode: {mode}")

    def _build_graph_self_items(self, cluster_items):
        n_clusters_all = len(cluster_items)
        try:
            assert n_clusters_all == self.n_clusters_all 
        except AssertionError:
            print(f"!!!!! n_clusters_all: {n_clusters_all}, self.n_clusters_all: {self.n_clusters_all}")
        
        edge_users, edge_items = [], [] 
        for i in range(n_clusters_all):
            for it in cluster_items[i]:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(n_clusters_all, self.n_items)
        )
        return UserItemNet, edge_users, edge_items

    def _build_graph_topk_cluster_items(self, sim_matrix, cluster_items):
        """
        cluster i connects to items covered by its Top-K similar clusters
        """
        n_clusters_all = len(cluster_items)
        try:
            assert n_clusters_all == self.n_clusters_all 
        except AssertionError:
            print(f"!!!!! n_clusters_all: {n_clusters_all}, self.n_clusters_all: {self.n_clusters_all}")
        
        topk_idx = self._select_topk_neighbors(sim_matrix)
        edge_users, edge_items = [], []

        for i in range(n_clusters_all):
            neigh_items = set()
            for j in topk_idx[i]:
                neigh_items |= cluster_items[j]
            for it in neigh_items:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(n_clusters_all, self.n_items)
        )
        return UserItemNet, edge_users, edge_items

    def _build_graph_self_and_topk_items(self, sim_matrix, cluster_items):
        n_clusters_all = len(cluster_items)
        try:
            assert n_clusters_all == self.n_clusters_all 
        except AssertionError:
            print(f"!!!!! n_clusters_all: {n_clusters_all}, self.n_clusters_all: {self.n_clusters_all}")
            
        topk_idx = self._select_topk_neighbors(sim_matrix)
        edge_users, edge_items = [], []
        for i in range(n_clusters_all):
            items = set(cluster_items[i])
            for j in topk_idx[i]:
                items |= cluster_items[j]
            for it in items:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(n_clusters_all, self.n_items)
        )
        return UserItemNet, edge_users, edge_items
    
    # =========================================================
    # Server 主流程
    # =========================================================
    def run(self, cluster_data_list):
        """
        cluster_data_list : List[dict]
                clustering.py 的输出
        return:
            updated_cluster_emb : dict
                key   : global_cluster_id
                value : torch.Tensor (updated embedding)
            updated_item_emb : Tensor, shape = (n_item, dim)
                server 端更新后的 item embedding
            server_rating : Tensor, shape = (n_clusters_all, n_items)
                server 端对所有 cluster 对所有 item 的评分矩阵
        """
        cluster_embs, cluster_items, cluster_keys = self._parse_clusters(cluster_data_list)
        n_clusters_all, dim = cluster_embs.shape
        try:
            assert n_clusters_all == self.n_clusters_all 
        except AssertionError:
            print(f"!!!!! n_clusters_all: {n_clusters_all}, self.n_clusters_all: {self.n_clusters_all}")

        # 获取 server 端 train_user_all 和 train_item_all，用于构建 server 端 dataset，从而构建 server 端 model
        if self.server_graph_enhance:
            sim_matrix = self._calc_cluster_sim(cluster_embs, cluster_items, mode="cosine")
            UserItemNet, train_user_all, train_item_all = self._build_server_graph(sim_matrix, cluster_items, mode="self+topk")
        else:
            UserItemNet, train_user_all, train_item_all = self._build_server_graph(None, cluster_items, mode="self")    
        # 构建 server 端 dataset
        server_dataset = dataloader.Loader(
            train_user_all=train_user_all,
            train_item_all=train_item_all,
            n_users=n_clusters_all,
            m_items=self.n_items,
            build_graph=True,
            device=self.device
        )
        # 构建 server 端 model
        server_recmodel = model.LightGCN(self.config, server_dataset)     # 模型简称映射到模型名
        server_recmodel = server_recmodel.to(self.device)
        bpr = utils.BPRLoss(server_recmodel, self.config)                                  
        if self.warm_start:
            server_recmodel.embedding_item.weight.data.copy_(self.aggregate_item_embeddings())
            server_recmodel.embedding_user.weight.data.copy_(cluster_embs)

        # 训练 server 端模型
        server_train_epochs = self.config['server_epochs']
        server_val_step = self.config['server_val_step']
        Neg_k = 1
        best_recall = -np.inf
        best_epoch = 0
        early_stop = self.config['server_early_stop']
        early_stop_cur = early_stop
        step = 5
        try:
            print(f"==================== Starting server training ====================")
            for epoch in range(1, server_train_epochs + 1):
                output_information = Procedure.BPR_train_original(
                    server_dataset,
                    server_recmodel,
                    bpr,
                    epoch,
                    neg_k=Neg_k,
                    isServer=True
                )
                print(f"Server EPOCH[{epoch}/{server_train_epochs}] {output_information}")
                # validation
                if epoch % server_val_step == 0:
                    cprint(f"[SERVER] [VALIDATION]")
                    recall = Procedure.Test(
                        server_dataset,
                        server_recmodel,
                        epoch,
                        multicore=self.config['multicore'],
                        test=0,        # 用 valDict
                        isServer=True
                    )
                    if recall > best_recall:
                        best_recall = recall
                        best_epoch = epoch
                        early_stop_cur = early_stop

                        weight_path = utils.getServerWeightFileName()
                        torch.save(server_recmodel.state_dict(), weight_path)
                    else:
                        early_stop_cur -= step
                        if early_stop_cur == 0:
                            break

        finally:
            print("===================== end the training of server ====================")
        print(f"Load best server model from epoch {best_epoch}")
        server_recmodel.load_state_dict(torch.load(weight_path, map_location=self.device))
        
        # 前向传播，得到更新后的 cluster embedding
        cluster_out, item_out = server_recmodel.computer()

        # 封装输出
        updated_cluster_emb = {
            key: emb.detach()
            for key, emb in zip(cluster_keys, cluster_out)
        }
        updated_item_emb = item_out.detach()

        print("===================== server test: cal server rating ====================")
        rating_out = Procedure.server_test(server_recmodel)
        server_rating = {
            key: rating
            for key, rating in zip(cluster_keys, rating_out)
        }

        return updated_cluster_emb, updated_item_emb, server_rating
