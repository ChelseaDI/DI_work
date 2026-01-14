import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix

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

    def __init__(self, config:dict, n_items, device="cpu"):
        """
            n_items : int
                总 item 数
            n_clusters_all : int
                全局 cluster 总数
            item_emb_buffer : list
                用于缓存各子图上传的 item embeddings
            topk : int
                每个 cluster 连接的最相似 cluster 数
            device : torch.device
        """
        self.device = device
        self.n_items = n_items
        self.config = config
        self.n_clusters_all = self.config['n_clusters_all']
        self.latent_dim = self.config['latent_dim_rec']
        self.topk = self.config['server_topk']
        self.warm_start = self.config['server_warm_start']
        self.server_graph_enhance = self.config['server_graph_enhance']

        self.item_emb_buffer = []
        self.__init_weight__()
        
    def __init_weight__(self):
        self.server_item_emb = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        self.embedding_cluster = torch.nn.Embedding(
            num_embeddings=self.n_clusters_all, embedding_dim=self.latent_dim) 
    
    def reset(self):
        """
        清空上一轮 server 聚合过程中缓存的状态
        """
        self.item_emb_buffer = []
        self.__init_weight__()
    
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
                cluster_embs.append(emb)
                cluster_items.append(set(data["cluster_items"][cid]))

        # (n_cluster, dim)
        cluster_embs = torch.tensor(
            np.stack(cluster_embs),
            dtype=torch.float32,
            device=self.device
        )
        return cluster_embs, cluster_items, cluster_keys
    
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
        sim_matrix = torch.zeros((self.n_clusters_all, self.n_clusters_all), device=self.device)

        for i in range(self.n_clusters_all):
            items_i = cluster_items[i]
            for j in range(i + 1, self.n_clusters_all):
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
    
    # 子图上传 item embedding
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
    
    def _select_topk_neighbors(self, sim_matrix):
        """
        Select Top-K similar clusters for each cluster.
        """
        # 取 Top-K 相似 cluster，构建 server-level 的 User–Item 图
        _, topk_idx = torch.topk(sim_matrix, self.topk, dim=1)      # topk_idx: (n_cluster, topk)
        return topk_idx
    
    def _build_server_graph(self, sim_matrix, cluster_items, mode="self"):
        if mode == "self":
            return self._build_graph_self_items(cluster_items)
        if mode == "topk":
            return self._build_graph_topk_cluster_items(sim_matrix, cluster_items)
        if mode == "self+topk":
            return self._build_graph_self_and_topk_items(sim_matrix, cluster_items)
        raise ValueError(f"Unknown graph mode: {mode}")

    def _build_graph_self_items(self, cluster_items):
        edge_users, edge_items = [], []

        for i in range(self.n_clusters_all):
            for it in cluster_items[i]:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(self.n_clusters_all, self.n_items)
        )
        return UserItemNet

    def _build_graph_topk_cluster_items(self, sim_matrix, cluster_items):
        """
        cluster i connects to items covered by its Top-K similar clusters
        """
        topk_idx = self._select_topk_neighbors(sim_matrix)
        
        edge_users, edge_items = [], []

        for i in range(self.n_clusters_all):
            neigh_items = set()
            for j in topk_idx[i]:
                neigh_items |= cluster_items[j]
            for it in neigh_items:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(self.n_clusters_all, self.n_items)
        )
        return UserItemNet
    
    def _build_graph_self_and_topk_items(self, sim_matrix, cluster_items):
        topk_idx = self._select_topk_neighbors(sim_matrix)

        edge_users, edge_items = [], []

        for i in range(self.n_clusters_all):
            items = set(cluster_items[i])   # 自身
            for j in topk_idx[i]:
                items |= cluster_items[j]   # 邻居扩散
            for it in items:
                edge_users.append(i)
                edge_items.append(it)

        UserItemNet = csr_matrix(
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(self.n_clusters_all, self.n_items)
        )
        return UserItemNet

    def _normalize_graph(self, UserItemNet):
        # 计算度（LightGCN 归一化用）
        user_deg = np.array(UserItemNet.sum(axis=1)).squeeze()
        item_deg = np.array(UserItemNet.sum(axis=0)).squeeze()
        user_deg[user_deg == 0] = 1
        item_deg[item_deg == 0] = 1
        # 构造对称归一化后的稀疏图（cluster + item）
        rows, cols = UserItemNet.nonzero()
        vals = 1.0 / np.sqrt(user_deg[rows] * item_deg[cols])
        idx_ci = torch.tensor([rows, cols + self.n_clusters_all], dtype=torch.long)         # 右上角    cluster -> item
        idx_ic = torch.tensor([cols + self.n_clusters_all, rows], dtype=torch.long)         # 左下角    item -> cluster
        indices = torch.cat([idx_ci, idx_ic], dim=1)
        values = torch.tensor(np.concatenate([vals, vals]), dtype=torch.float32)

        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(self.n_clusters_all + self.n_items, self.n_clusters_all + self.n_items),
            device=self.device
        )

    def run(self, cluster_data_list):
        """
        cluster_data_list : List[dict]
                clustering.py 的输出
        return:
            updated_cluster_emb : dict
                key   : global_cluster_id
                value : torch.Tensor (updated embedding)
        """
        cluster_embs, cluster_items, cluster_keys = self._parse_clusters(cluster_data_list)
        n_cluster, dim = cluster_embs.shape

        # 构建 server-level User–Item 图
        if self.server_graph_enhance:
            sim_matrix = self._calc_cluster_sim(cluster_embs, cluster_items, mode="cosine")
            UserItemNet = self._build_server_graph(sim_matrix, cluster_items, mode="self+topk")
        else:
            UserItemNet = self._build_server_graph(None, cluster_items, mode="self")    
        # 图归一化
        Graph = self._normalize_graph(UserItemNet)                                      

        if self.warm_start:
            self.server_item_emb.weight.data.copy_(self.aggregate_item_embeddings())
            self.embedding_cluster.weight.data.copy_(cluster_embs)

        # 实例化 server LightGCN
        server_lgn = ServerLightGCN(
            n_cluster=n_cluster,
            n_item=self.n_items,
            dim=dim,
            n_layers=3,
            Graph=Graph
        ).to(self.device)
        # 初始化 embedding
        server_lgn.embedding_cluster.weight.data.copy_(self.embedding_cluster.weight.data)
        server_lgn.embedding_item.weight.data.copy_(self.server_item_emb.weight.data)
        # 前向传播，得到更新后的 cluster embedding
        cluster_out, item_out = server_lgn.computer()

        # 封装输出
        updated_cluster_emb = {
            key: emb.detach()
            for key, emb in zip(cluster_keys, cluster_out)
        }
        updated_item_emb = item_out.detach()

        return updated_cluster_emb, updated_item_emb

# ======================================================
# Server-side LightGCN 对象
# ======================================================
class ServerLightGCN(torch.nn.Module):
    def __init__(self, n_cluster, n_item, dim, n_layers, Graph):
        super().__init__()
        self.n_cluster = n_cluster
        self.n_item = n_item
        self.n_layers = n_layers
        self.embedding_cluster = torch.nn.Embedding(n_cluster, dim)
        self.embedding_item = torch.nn.Embedding(n_item, dim)
        self.Graph = Graph
    def computer(self):
        cluster_emb = self.embedding_cluster.weight
        item_emb = self.embedding_item.weight
        all_emb = torch.cat([cluster_emb, item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)
        return torch.split(out, [self.n_cluster, self.n_item])