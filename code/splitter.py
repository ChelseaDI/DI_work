import numpy as np
from scipy.sparse import csr_matrix
import world   # 你的全局配置模块
import torch
import scipy.sparse as sp
import os
from os.path import join

class SubgraphSplitter:
    """
    将完整 Loader 拆成若干子图 Loader
    用法：
        splitter = SubgraphSplitter(group_size=200, seed=world.seed)
        sub_loaders = splitter.split(loader)
    """

    def __init__(self, group_size=200, seed=42):
        self.group_size = group_size
        self.seed = seed

    # ---------------- 对外唯一接口 ----------------
    def split(self, full_loader):
        """
        full_loader : 已经完整初始化好的 Loader 实例
        return      : List[Loader]，每个元素是一个子图 loader
        """
        user_groups = self._split_user_groups(full_loader)
        sub_loaders = []
        for gid, users in enumerate(user_groups):
            sub_loaders.append(self._build_one_subloader(full_loader, gid, users))
        return sub_loaders

    # ---------------- 内部辅助 ----------------
    # def _split_user_groups(self, loader):
    #     """按 group_size 把用户随机切成若干组"""
    #     users = loader.trainUniqueUsers.copy()
    #     rng = np.random.RandomState(self.seed)
    #     rng.shuffle(users)
    #     n_group = int(np.ceil(len(users) / self.group_size))
    #     groups = []
    #     for g in range(n_group):
    #         start = g * self.group_size
    #         end = min((g + 1) * self.group_size, len(users))
    #         groups.append(users[start:end])
    #     return groups

    def _split_user_groups(self, loader):
        """按 group_size 把用户随机切成若干组（支持落盘复用）"""
        # ===============================
        # 分组文件路径
        # ===============================
        save_dir = os.path.join(loader.path, "user_groups")
        os.makedirs(save_dir, exist_ok=True)
        group_file = os.path.join(save_dir, f"groups_size{self.group_size}_seed{self.seed}.npy")
        # ===============================
        # 如果文件存在，直接加载
        # ===============================
        if os.path.exists(group_file):
            print(f"[SubgraphSplitter] Load user groups from {group_file}")
            groups = np.load(group_file, allow_pickle=True)
            return list(groups)
        # ===============================
        # 否则重新随机划分
        # ===============================
        print(f"[SubgraphSplitter] Split users and save to {group_file}")
        users = loader.trainUniqueUsers.copy()
        rng = np.random.RandomState(self.seed)
        rng.shuffle(users)
        n_group = int(np.ceil(len(users) / self.group_size))
        groups = []
        for g in range(n_group):
            start = g * self.group_size
            end = min((g + 1) * self.group_size, len(users))
            groups.append(users[start:end])
        # ===============================
        # 保存到文件
        # ===============================
        np.save(group_file, np.array(groups, dtype=object), allow_pickle=True)
        return groups


    def _build_one_subloader(self, full_loader, gid, users):
        """
        为单独一组用户构造一个 Loader 子图对象
            1. 组内训练交互
            2. 组内测试交互
            3. 对应的 UserItemNet 稀疏矩阵
            4. 其他必要字段，保证后续训练代码能直接跑
        """
        users = list(set(users))
        users = sorted(users)                       # 保证顺序一致
        # global_uid -> local_uid
        uid_map = {g_uid: l_uid for l_uid, g_uid in enumerate(users)}
        # uid_rev[local_uid] = global_uid
        uid_rev = users

        # 训练集
        tr_mask = np.isin(full_loader.trainUser, users)
        g_tr_user = np.array([uid_map[g_uid] for g_uid in full_loader.trainUser[tr_mask]])
        g_tr_item = full_loader.trainItem[tr_mask]
        # ====== 【新增】train_all ======
        tr_all_mask = np.isin(full_loader.trainUser_all, users)
        g_tr_user_all = np.array([uid_map[g_uid] for g_uid in full_loader.trainUser_all[tr_all_mask]])
        g_tr_item_all = full_loader.trainItem_all[tr_all_mask]
        # 验证集
        va_mask = np.isin(full_loader.valUser, users)
        g_va_user = np.array([uid_map[g_uid] for g_uid in full_loader.valUser[va_mask]])
        g_va_item = full_loader.valItem[va_mask]
        # 测试集
        te_mask = np.isin(full_loader.testUser, users)
        g_te_user = np.array([uid_map[g_uid] for g_uid in full_loader.testUser[te_mask]])
        g_te_item = full_loader.testItem[te_mask]

        # 新建空 Loader，把必要字段填进去 
        sub = full_loader.__class__.__new__(full_loader.__class__)  # 不触发 __init__
        
        # 组编号
        sub.group_id = gid

        # 基本配置 
        sub.split      = full_loader.split
        sub.folds      = full_loader.folds
        sub.mode_dict  = full_loader.mode_dict
        sub.mode       = full_loader.mode
        sub.path       = full_loader.path + "/group_" + str(sub.group_id)
        sub.Graph      = None

        sub.n_user = len(users)
        sub.m_item = full_loader.m_items
        # 训练数据 
        sub.trainUser        = g_tr_user
        sub.trainItem        = g_tr_item
        sub.trainUser_all = g_tr_user_all          # 【新增】
        sub.trainItem_all = g_tr_item_all          # 【新增】
        sub.trainUniqueUsers = np.arange(sub.n_user)   # 局部 0..n-1
        sub.traindataSize    = len(g_tr_user)
        # 测试数据 
        sub.testUser        = g_te_user
        sub.testItem        = g_te_item
        sub.testUniqueUsers = np.unique(g_te_user)
        sub.testDataSize    = len(g_te_user)
        # ====== 【新增】验证数据 ======
        sub.valUser        = g_va_user
        sub.valItem        = g_va_item
        sub.valUniqueUsers = np.unique(g_va_user)
        sub.valDataSize    = len(g_va_user)

        # UserItemNet 子图稀疏矩阵 
        sub.UserItemNet = csr_matrix(
            (np.ones(len(sub.trainUser)), (sub.trainUser, sub.trainItem)),
            shape=(sub.n_user, sub.m_item))
        sub.UserItemNet_all = csr_matrix(
            (np.ones(len(sub.trainUser_all)), (sub.trainUser_all, sub.trainItem_all)),
            shape=(sub.n_user, sub.m_item))

        sub.users_D = np.array(sub.UserItemNet.sum(axis=1)).squeeze()     # (n_user, 1) squeeze-> (n_user,) 用户交互物品数量
        sub.users_D[sub.users_D == 0.] = 1
        sub.items_D = np.array(sub.UserItemNet.sum(axis=0)).squeeze()
        sub.items_D[sub.items_D == 0.] = 1.
        
        # pre-calculate
        # print(f"@@@@@@@@@@@@@@@@@@@@@@@ [SubgraphSplitter] Pre-calculate subloader for group {gid} @@@@@@@@@@@@@@@@@@@@@@@")
        sub._allPos = sub.getUserPosItems(list(range(sub.n_user)))
        sub._Loader__testDict = sub._Loader__build_test()
        # print("----- subsubsubsubsubsubsubsubsubsubsubsub ----")
        sub._Loader__valDict = sub._Loader__build_val()

        # # 用户正样本列表
        # sub._allPos = [[] for _ in range(sub.n_user)]
        # for u in sub.trainUniqueUsers:
        #     sub._allPos[u] = sub.UserItemNet[u].nonzero()[1].astype(np.int64)
        
        # # testDict 
        # sub._Loader__testDict = {}
        # for u, i in zip(g_te_user, g_te_item):
        #     sub._Loader__testDict.setdefault(u, []).append(i)
        # # ====== 【新增】valDict ======
        # sub._Loader__valDict = {}
        # for u, i in zip(g_va_user, g_va_item):
        #     sub._Loader__valDict.setdefault(u, []).append(i)
        
        print(f"[Group {gid}] "
              f"train={sub.traindataSize}, "
              f"val={sub.valDataSize}, "
              f"test={sub.testDataSize}")

        return sub

class SubLoaderUtils:
    """静态工具类，挂到子 Loader 上，方便调用"""
    @staticmethod
    def to_local(sub, g_uid):
        """全局 uid -> 局部 uid"""
        return sub.uid_map[g_uid]

    @staticmethod
    def to_global(sub, l_uid):
        """局部 uid -> 全局 uid"""
        return sub.uid_rev[l_uid]