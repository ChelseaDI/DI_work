"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint
from time import time

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            # user_dim = torch.LongTensor(self.trainUser)
            # item_dim = torch.LongTensor(self.trainItem)
            user_dim = torch.tensor(self.trainUser, dtype=torch.long)
            item_dim = torch.tensor(self.trainItem, dtype=torch.long)
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)

class Loader(BasicDataset):
    """
    ===========================
    Abstract Interaction Loader
    ===========================

    设计目标：
    1. 不负责文件 IO
    2. 不负责 train / val / test 的切分
    3. 只依赖「已准备好的交互数据」
    4. 同时支持：
        - full loader
        - sub loader
        - server loader
        - BPR / LightGCN
    
    注意：
    - UserItemNet       : train-only（训练 & LightGCN）
    - UserItemNet_all   : train + val + test（仅用于 eval mask）
    """

    def _divide_val(self,train_user_all,train_item_all):
        self.trainUser_all = np.asarray(train_user_all, dtype=np.int64)
        self.trainItem_all = np.asarray(train_item_all, dtype=np.int64)

        # Divide train and validation
        trainData = pd.DataFrame({'user':train_user_all, 'item':train_item_all})          # 用户，物品 对
        val = trainData.sample(frac=0.1, replace=False, random_state=2022)      # 从训练集中抽取 10% 作为验证集（validation set）
        val.sort_index(inplace=True)
        trainData.drop(val.index, inplace=True)

        self.trainUser = trainData['user'].values
        self.trainItem = trainData['item'].values
        self.valUser = val['user'].values
        self.valItem = val['item'].values

    def __init__(
        self,
        train_user_all=None,
        train_item_all=None,
        train_user=None,
        train_item=None,
        n_users=None,
        m_items=None,
        val_user=None,
        val_item=None,
        test_user=None,
        test_item=None,
        build_graph=True,
        graph_split=False,
        n_fold=100,
        device="cpu",
        cache_path=None,
    ):
        """
        Parameters
        ----------
        train_user / train_item : array-like
            训练集中的 (user, item) 交互对
        n_users / m_items : int
            全局用户数 / 物品数（非常重要，用于 sub / server）
        val_user / val_item : optional
            验证集交互
        test_user / test_item : optional
            测试集交互
        build_graph : bool
            是否构建 LightGCN 所需的图
        graph_split : bool
            是否对邻接矩阵进行 fold 切分
        n_fold : int
            邻接矩阵切分份数
        device : torch.device or str
        cache_path : str or None
            若不为 None，则缓存归一化邻接矩阵
        """

        super().__init__()

        # ========= 1. 基础交互数据 =========
        if train_user_all is not None and train_item_all is not None:
            self._divide_val(train_user_all,train_item_all)
        else:
            self.trainUser = np.asarray(train_user, dtype=np.int64)
            self.trainItem = np.asarray(train_item, dtype=np.int64)
            self.valUser = np.asarray(val_user, dtype=np.int64) 
            self.valItem = np.asarray(val_item, dtype=np.int64) 
            # train_user_all = train_user + val_user
            # train_item_all = train_item + val_item
            # self.trainUser_all = np.asarray(train_user_all, dtype=np.int64)
            # self.trainItem_all = np.asarray(train_item_all, dtype=np.int64)
            self.trainUser_all = np.concatenate([train_user, val_user]).astype(np.int64)
            self.trainItem_all = np.concatenate([train_item, val_item]).astype(np.int64)



        assert len(self.trainUser) == len(self.trainItem), \
            "train_user and train_item length mismatch"

        # 用户 / 物品规模
        self.n_user = n_users if n_users is not None else int(self.trainUser.max() + 1)
        self.m_item = m_items if m_items is not None else int(self.trainItem.max() + 1)
        self.traindataSize = len(self.trainUser)

        # ========= 2. 验证 / 测试（可选） =========
        self.testUser = np.asarray(test_user, dtype=np.int64) if test_user is not None else None
        self.testItem = np.asarray(test_item, dtype=np.int64) if test_item is not None else None

        self._valDict = self._build_dict(self.valUser, self.valItem)
        self._testDict = self._build_dict(self.testUser, self.testItem)

        # ========= 3. User-Item 稀疏矩阵 =========
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                        shape=(self.n_user, self.m_item))       # 稀疏矩阵（值，坐标，矩阵大小）
        self.UserItemNet_all = csr_matrix((np.ones(len(self.trainUser_all)), (self.trainUser_all, self.trainItem_all)),
                                        shape=(self.n_user, self.m_item)) 

        # 每个用户 / 物品的度（LightGCN 用）
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1.

        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.

        # ========= 4. 正样本缓存（BPR / eval） =========
        self._allPos = self.getUserPosItems(range(self.n_user))

        # ========= 5. 图相关配置 =========
        self.build_graph = build_graph
        self.graph_split = graph_split
        self.n_fold = n_fold
        self.device = device
        self.cache_path = cache_path
        self.Graph = None

        print(f"[Loader] n_users={self.n_user}, m_items={self.m_item}")
        print(f"[Loader] train interactions={self.traindataSize}")
        if self.valUser is not None:
            print(f"[Loader] val interactions={len(self.valUser)}")
        if self.testUser is not None:
            print(f"[Loader] test interactions={len(self.testUser)}")

    # ======================================================
    # Properties（与你原 Loader 对齐，保证兼容）
    # ======================================================
    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos

    @property
    def testDict(self):
        return self._testDict

    @property
    def valDict(self):
        return self._valDict

    # ======================================================
    # Graph Construction（LightGCN）
    # ======================================================
    def getSparseGraph(self):
        """
        延迟构建归一化邻接矩阵
        """
        if not self.build_graph:
            return None

        if self.Graph is not None:
            return self.Graph

        print("[Loader] Building normalized adjacency matrix...")
        t = time()

        # ---------- 获取 norm_adj ----------
        norm_adj = self._get_norm_adj_mat()

        # ---------- split or not ----------
        if self.graph_split:
            self.Graph = self._split_A_hat(norm_adj)
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)

        print(f"[Loader] Graph ready, time={time() - t:.2f}s")
        return self.Graph

    def _get_norm_adj_mat(self):
        if self.cache_path is not None:
            os.makedirs(self.cache_path, exist_ok=True)
            file_path = os.path.join(self.cache_path, 's_pre_adj_mat.npz')
            if os.path.exists(file_path):
                norm_adj = sp.load_npz(file_path)
                print("[Loader] Loaded cached adjacency matrix")
                return norm_adj
            else:
                norm_adj = self._build_norm_adj()
                sp.save_npz(file_path, norm_adj)
                print("[Loader] Saved adjacency matrix to cache")
                return norm_adj
        else:
            norm_adj = self._build_norm_adj()
            print("[Loader] Built adjacency matrix, not cached !")    
            return norm_adj
        
    def _build_norm_adj(self):
        """
        构建 LightGCN 使用的 D^{-1/2} A D^{-1/2}
        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.m_items, self.n_users + self.m_items),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()

        R = self.UserItemNet.tolil()
        adj_mat[:self.n_users, self.n_users:] = R       # 右上角
        adj_mat[self.n_users:, :self.n_users] = R.T     # 左下角

        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        return d_mat.dot(adj_mat).dot(d_mat).tocsr()

    def _split_A_hat(self, A):
        """
        将邻接矩阵切分成多个 fold（显存友好）
        """
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.n_fold

        for i in range(self.n_fold):
            start = i * fold_len
            end = (self.n_users + self.m_items) if i == self.n_fold - 1 else (i + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device)
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = torch.LongTensor([coo.row, coo.col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, data, torch.Size(coo.shape))

    # ======================================================
    # Utility Functions（BPR / Eval）
    # ======================================================
    def _build_dict(self, users, items):
        if users is None or items is None:
            return None
        d = {}
        for u, i in zip(users, items):
            d.setdefault(int(u), []).append(int(i))
        return d

    def getUserItemFeedback(self, users, items):
        """
        判断 (u, i) 是否为正样本
        """
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape(-1)

    def getUserPosItems(self, users):
        """
        返回每个用户的正样本 item 列表
        """
        posItems = []
        for u in users:
            posItems.append(self.UserItemNet[u].nonzero()[1])
        return posItems
    
    def getUserPosItems_Test(self, users):
        """
        [COMPAT] 返回用户在“全历史交互”中的正样本
        用于 test / val 阶段的负采样屏蔽（兼容旧 procedure）
        """
        posItems = []
        for u in users:
            posItems.append(self.UserItemNet_all[u].nonzero()[1])
        return posItems
