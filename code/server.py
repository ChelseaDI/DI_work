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

    def __init__(self, n_item, topk=5, device="cpu"):
        """
            n_item : int
                总 item 数
            item_emb_buffer : list
                用于缓存各子图上传的 item embeddings
            topk : int
                每个 cluster 连接的最相似 cluster 数
            device : torch.device
        """
        self.n_item = n_item
        self.item_emb_buffer = []
        self.topk = topk
        self.device = device
        self.server_item_emb = None

    def reset(self):
        """
        清空上一轮 server 聚合过程中缓存的状态
        """
        self.item_emb_buffer = []
        self.server_item_emb = None

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
    
    def get_global_item_embedding(self):
        assert self.server_item_emb is not None
        return self.server_item_emb


    def run(self, cluster_data_list):
        """
        cluster_data_list : List[dict]
                clustering.py 的输出
        return:
            updated_cluster_emb : dict
                key   : global_cluster_id
                value : torch.Tensor (updated embedding)
        """
        # 用于将各 group 的 clusters data 平铺
        # [ [group1_cluster1], [group1_cluster2], [group2_cluster1], ... ]
        cluster_embs = []
        cluster_items = []
        cluster_keys = []   # 用于索引 cluster

        for data in cluster_data_list:      # 遍历每组的 clusters
            gid = data["group_id"]
            for cid, emb in data["cluster_embeddings"].items():
                key = (gid, cid)
                cluster_keys.append(key)
                cluster_embs.append(emb)
                cluster_items.append(set(data["cluster_items"][cid]))

        # (N_cluster, dim)
        cluster_embs = torch.tensor(
            np.stack(cluster_embs),         # torch.stack?
            dtype=torch.float32,
            device=self.device
        )  
        n_cluster, dim = cluster_embs.shape

        # # 计算 cluster 间相似度
        # norm_emb = F.normalize(cluster_embs, dim=1)            # 对张量按行归一化  a·b = ||a||·||b||·cos(θ) = cos(θ)
        # sim_matrix = torch.matmul(norm_emb, norm_emb.t())      # (n_cluster, n_cluster)
        # sim_matrix.fill_diagonal_(-1e9)                        # 自己对自己 -- 负无穷

        # ======================================================
        # 基于 cluster_items 的 Jaccard 相似度
        # ======================================================
        sim_matrix = torch.zeros((n_cluster, n_cluster), device=self.device)
        for i in range(n_cluster):
            items_i = cluster_items[i]
            for j in range(i + 1, n_cluster):
                items_j = cluster_items[j]
                inter = len(items_i & items_j)
                if inter == 0:
                    continue
                union = len(items_i | items_j)
                sim = inter / union
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        sim_matrix.fill_diagonal_(-1e9)


        # 取 Top-K 相似 cluster，构建 server-level 的 User–Item 图
        _, topk_idx = torch.topk(sim_matrix, self.topk, dim=1)      # topk_idx: (n_cluster, topk)
        edge_users = []
        edge_items = []
        for i in range(n_cluster):
            neigh_ids = topk_idx[i]
            neigh_items = set()
            for j in neigh_ids:
                neigh_items |= cluster_items[j]     # 并集
            for it in neigh_items:                  # 与 Top-K 相似 cluster 覆盖的所有 item 间建立边
            # for it in cluster_items[i]:
                edge_users.append(i)
                edge_items.append(it)
        edge_users = np.array(edge_users)
        edge_items = np.array(edge_items)
        UserItemNet = csr_matrix(                   # 构造 user–Item 图 (n_cluster, n_item)
            (np.ones(len(edge_users)), (edge_users, edge_items)),
            shape=(n_cluster, self.n_item)
        )                                           

        # item_embeddings = self.aggregate_item_embeddings()   # (n_item, dim)
        self.server_item_emb = self.aggregate_item_embeddings()
        item_embeddings = self.server_item_emb

        # 计算度（LightGCN 归一化用）
        user_deg = np.array(UserItemNet.sum(axis=1)).squeeze()
        item_deg = np.array(UserItemNet.sum(axis=0)).squeeze()
        user_deg[user_deg == 0] = 1
        item_deg[item_deg == 0] = 1
        # 构造对称归一化后的稀疏图（cluster + item）
        rows, cols = UserItemNet.nonzero()
        vals = np.ones(len(rows), dtype=np.float32)
        norm_vals = vals / np.sqrt(user_deg[rows] * item_deg[cols])
        # idx_ci = torch.LongTensor([rows, cols + n_cluster])         # 右上角    cluster -> item
        # idx_ic = torch.LongTensor([cols + n_cluster, rows])         # 左下角    item -> cluster
        idx_ci = torch.tensor([rows, cols + n_cluster], dtype=torch.long)        # 右上角    cluster -> item
        idx_ic = torch.tensor([cols + n_cluster, rows], dtype=torch.long)         # 左下角    item -> cluster
        indices = torch.cat([idx_ci, idx_ic], dim=1)
        values = torch.FloatTensor(np.concatenate([norm_vals, norm_vals]))
        Graph = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(n_cluster + self.n_item, n_cluster + self.n_item),
            device=self.device
        )

        # 实例化 server LightGCN
        server_lgn = ServerLightGCN(
            n_cluster=n_cluster,
            n_item=self.n_item,
            dim=dim,
            n_layers=3,
            Graph=Graph
        ).to(self.device)
        # 初始化 embedding
        server_lgn.embedding_cluster.weight.data.copy_(cluster_embs)
        server_lgn.embedding_item.weight.data.copy_(item_embeddings)
        # 前向传播，得到更新后的 cluster embedding
        cluster_out, _ = server_lgn.computer()
        # 封装输出
        updated_cluster_emb = {
            key: emb
            for key, emb in zip(cluster_keys, cluster_out)
        }

        return updated_cluster_emb

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