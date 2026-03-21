import world
import dataloader
import model
import utils
from pprint import pprint

import splitter
from world import cprint


# =====================================================
# 1. 读取原始交互（只做 IO，不构造 Loader）
# =====================================================
def load_interactions(dataset_name, data_path):
    """
    统一的数据读取入口
    返回：
        train_user, train_item
        test_user, test_item
        n_users, m_items
    """
    train_user, train_item = [], []
    test_user, test_item = [], []

    train_file = f"{data_path}/train.txt"
    test_file = f"{data_path}/test.txt"

    n_user, m_item = 0, 0

    with open(train_file) as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            u = int(parts[0])
            items = list(map(int, parts[1:]))

            train_user.extend([u] * len(items))
            train_item.extend(items)

            n_user = max(n_user, u)
            m_item = max(m_item, max(items))

    with open(test_file) as f:
        # ppp = 1
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            u = int(parts[0])
            # print(f"行数：{ppp}\n")
            # ppp = ppp + 1
            items = list(map(int, parts[1:]))

            test_user.extend([u] * len(items))
            test_item.extend(items)

            n_user = max(n_user, u)
            if items:
                m_item = max(m_item, max(items))

    return (
        train_user,
        train_item,
        test_user,
        test_item,
        n_user + 1,
        m_item + 1,
    )


# =====================================================
# 2. 构造 Full Dataset（全局）
# =====================================================
if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:

    cprint(f"[REGISTER] Loading dataset {world.dataset}")

    data_path = f"../data/{world.dataset}"

    (
        train_user_all,
        train_item_all,
        test_user,
        test_item,
        n_users,
        m_items,
    ) = load_interactions(world.dataset, data_path)

    # ---------- Full Loader ----------
    dataset = dataloader.Loader(
        train_user_all=train_user_all,
        train_item_all=train_item_all,
        test_user=test_user,
        test_item=test_item,
        n_users=n_users,
        m_items=m_items,
        build_graph=True,
        graph_split=world.config['A_split'],
        n_fold=world.config['A_n_fold'],
        device=world.device,
        cache_path=data_path,
    )

    # 全局物品数（后面 server / model 都要用）
    m_item = dataset.m_items

    # =================================================
    # 3. 构造 Sub Datasets（通过 Splitter）
    # =================================================
    group_size = world.config['group_size']
    seed = world.seed

    splitter_obj = splitter.SubgraphSplitter(
        group_size=group_size,
        seed=seed,
        cache_groups=True       # 默认会缓存分组结果
    )

    sub_datasets = splitter_obj.split(dataset)

elif world.dataset == 'lastfm':
    raise NotImplementedError("lastfm loader not refactored yet")


# =====================================================
# 4. 打印配置（保持你原来的逻辑）
# =====================================================
print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')


# =====================================================
# 5. Model Registry（不变）
# =====================================================
MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
