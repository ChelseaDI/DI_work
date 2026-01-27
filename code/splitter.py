import os
import numpy as np
from typing import List, Dict, Tuple
from dataloader import Loader


class SubgraphSplitter:
    """
    将 full Loader 按用户拆分为多个 sub Loader
    """

    def __init__(self, group_size=200, seed=42, cache_groups=True):
        self.group_size = group_size
        self.seed = seed
        self.cache_groups = cache_groups

    # ======================================================
    # Public API
    # ======================================================
    def split(self, full_loader: Loader) -> List[Loader]:
        """
        主入口：将 full loader 拆分成多个 sub loader
        """
        user_groups = self._get_user_groups(full_loader)
        sub_loaders = []

        for gid, users in enumerate(user_groups):
            sub_loader = self._build_subloader(full_loader, gid, users)
            sub_loaders.append(sub_loader)

        return sub_loaders

    # ======================================================
    # Step 1: 用户分组
    # ======================================================
    def _get_user_groups(self, loader: Loader) -> List[np.ndarray]:
        """
        按 group_size 随机划分用户（支持落盘复用）
        """
        if not self.cache_groups:
            return self._split_users(loader)

        save_dir = os.path.join(loader.cache_path or ".", "user_groups")
        os.makedirs(save_dir, exist_ok=True)

        file = os.path.join(
            save_dir,
            f"groups_size{self.group_size}_seed{self.seed}.npy"
        )

        if os.path.exists(file):
            print(f"[Splitter] Load user groups from {file}")
            return list(np.load(file, allow_pickle=True))

        groups = self._split_users(loader)
        np.save(file, np.array(groups, dtype=object))
        print(f"[Splitter] Save user groups to {file}")
        return groups

    def _split_users(self, loader: Loader) -> List[np.ndarray]:
        users = np.arange(loader.n_users)
        rng = np.random.RandomState(self.seed)
        rng.shuffle(users)

        n_group = int(np.ceil(len(users) / self.group_size))
        return [
            users[i * self.group_size: (i + 1) * self.group_size]
            for i in range(n_group)
        ]

    # ======================================================
    # Step 2: 构造单个 sub loader
    # ======================================================
    def _build_subloader(
        self,
        full: Loader,
        gid: int,
        users: np.ndarray
    ) -> Loader:
        """
        为一组用户构造标准 Loader
        """
        users = np.unique(users)
        users.sort()

        uid_map = self._build_uid_map(users)

        train_u, train_i = self._filter_interactions(
            full.trainUser, full.trainItem, uid_map
        )
        val_u, val_i = self._filter_interactions(
            full.valUser, full.valItem, uid_map
        )
        test_u, test_i = self._filter_interactions(
            full.testUser, full.testItem, uid_map
        )

        sub = Loader(
            train_user=train_u,
            train_item=train_i,
            val_user=val_u,
            val_item=val_i,
            test_user=test_u,
            test_item=test_i,
            n_users=len(users),
            m_items=full.m_items,
            build_graph=full.build_graph,
            graph_split=full.graph_split,
            n_fold=full.n_fold,
            device=full.device,
            cache_path=f"{full.cache_path}/group_{gid}",      # sub graph 缓存
        )

        # -------- 附加元信息（不影响 Loader 逻辑）--------
        sub.group_id = gid
        sub.global_users = users
        sub.uid_map = uid_map
        sub.item_set = self._cal_item_set(sub)

        print(
            f"[Group {gid}] "
            f"users={len(users)}, "
            f"item_set={len(sub.item_set)}, "
            f"train={sub.trainDataSize}, "
            f"val={0 if val_u is None else len(val_u)}, "
            f"test={0 if test_u is None else len(test_u)}"
        )

        return sub

    # ======================================================
    # Utility
    # ======================================================
    def _build_uid_map(self, users: np.ndarray) -> Dict[int, int]:
        """
        global uid -> local uid
        """
        return {int(u): i for i, u in enumerate(users)}

    def _filter_interactions(
        self,
        user_arr,
        item_arr,
        uid_map: Dict[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 (user, item) 中筛选属于该 subgraph 的交互
        """
        if user_arr is None or item_arr is None:
            return None, None

        mask = np.isin(user_arr, list(uid_map.keys()))
        users = user_arr[mask]
        items = item_arr[mask]

        local_users = np.array([uid_map[int(u)] for u in users], dtype=np.int64)
        return local_users, items
    
    def _cal_item_set(self, subLoader:Loader) -> Dict[int, set]:
        item_set = set()
        for items in subLoader.getUserPosItems_Test(np.arange(subLoader.n_users)):
            item_set.update(items)   # train + val
        return item_set
