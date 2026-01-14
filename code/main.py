import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import os                                         
from os.path import join
from collections import defaultdict
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import m_item
from register import sub_datasets

import clustering
from clustering import UserClustering
clustering = UserClustering(
    n_clusters=world.config['n_clusters'],
    seed=world.seed
)

from server import ServerGraph
server_graph = ServerGraph (
    config=world.config,
    n_items = m_item,   # 全局 item 数
    device=world.device,
)

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

Neg_k = 1
# >>>>>>>>> 【新增】validation & early stop 配置 <<<<<<<<<
early_stop = world.config.get("early_stop", 50)
val_step = world.config.get("val_step", 5)

group_best_recall = {}
group_best_epoch = {}
group_early_stop_cur = {}

# ======================================================
# ===== Stage 0: 初始化每个 group 的模型（只做一次）=====
# ======================================================
group_models = {}          # >>> NEW
group_bpr = {}             # >>> NEW
group_datasets = {}        # >>> NEW

# ======================================================
# group-level training and testing，然后 clustering （基于 k-means）
# ======================================================
for dataset in sub_datasets:
    gid = dataset.group_id
    print(f"[INIT] Group {gid}")

    # >>>>>>>>> 【新增】每个 group 的 validation 状态 <<<<<<<<<
    group_best_recall[gid] = -np.inf
    group_best_epoch[gid] = 0
    group_early_stop_cur[gid] = early_stop

    Recmodel = register.MODELS[world.model_name](world.config, dataset)     # 模型简称映射到模型名
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    weight_file = utils.getFileName(Recmodel)
    print(f"load and save to {weight_file}")
    os.makedirs(os.path.dirname(weight_file), exist_ok=True)

    # group-level training and testing
    if os.path.exists(weight_file) and world.config['load_pretrain']:
        print(f"[Group {dataset.group_id}] : Load pretrained results")
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        world.cprint(f"loaded model weights from {weight_file}")
    else:
        print(f"[Group {dataset.group_id}] : Training LightGCN from scratch")
        if world.LOAD:
            print(f"!!!!!! try to carry on training from {weight_file} !!!!!!")
            try:
                Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
                world.cprint(f"loaded model weights from {weight_file}")
            except FileNotFoundError:
                print(f"{weight_file} not exists, start from beginning")
        try:
            # >>>>>>>>> 【修改】加入 validation + early stopping <<<<<<<<<
            for epoch in range(1, world.TRAIN_epochs + 1):
                output_information = Procedure.BPR_train_original(
                    dataset,
                    Recmodel,
                    bpr,
                    epoch,
                    neg_k=Neg_k,
                    w=w
                )
                print(f"[Group {gid}] EPOCH[{epoch}/{world.TRAIN_epochs}] {output_information}")

                # ===== Validation =====
                if epoch % val_step == 0:
                    cprint(f"[Group {gid}] [VALIDATION]")
                    recall = Procedure.Test(
                        dataset,
                        Recmodel,
                        epoch,
                        w,
                        world.config['multicore'],
                        test=0        # <<< 用 valDict
                    )

                    if recall > group_best_recall[gid]:
                        group_best_recall[gid] = recall
                        group_best_epoch[gid] = epoch
                        group_early_stop_cur[gid] = early_stop

                        torch.save(Recmodel.state_dict(), weight_file)
                        print(f"[Group {gid}] New best val recall = {recall:.6f}, model saved")
                    else:
                        group_early_stop_cur[gid] -= val_step
                        if group_early_stop_cur[gid] <= 0:
                            print(f"[Group {gid}] Early stopping at epoch {epoch}")
                            break

        finally:
            print(f"!!! end the training of group {dataset.group_id} !!!")
            if world.tensorboard:
                w.close()

        # >>>>>>>>> 【修改】加载 validation 最优模型 <<<<<<<<<
        print(f"[Group {gid}] Load best model from epoch {group_best_epoch[gid]}")
        Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))

        print(f"[Group {gid}] [FINAL TEST]")
        Procedure.Test(
            dataset,
            Recmodel,
            group_best_epoch[gid],
            w,
            world.config['multicore'],
            test=1        # <<< 用 testDict
        )
    
    # 缓存模型
    group_models[gid] = Recmodel
    group_bpr[gid] = bpr
    group_datasets[gid] = dataset
print("================ Initialization Finished ================")

# global-local 多轮迭代 初始化
global_rounds = world.config.get("global_rounds", 3)          # 总轮次
local_epochs = world.config.get("local_epochs", 5)            # user emb 更新后，group-level training 轮次
alpha = world.config.get("cluster_alpha", 0.1)                # user emb 学习率

for r in range(global_rounds):
    print(f"\n================ Global Round {r} ================")
    cluster_data_list = []   # 收集该轮次所有 group 的 cluster 信息
    server_graph.reset()
    for gid, Recmodel in group_models.items():
        # clustering
        dataset = group_datasets[gid]
        cluster_data = clustering.run(dataset, Recmodel)
        cluster_data_list.append(cluster_data)          # [ [group0的clusters], [group1的clusters], [group2的clusters],...]
        # server collecting
        _, item_emb = Recmodel.computer()
        server_graph.collect_item_embeddings(item_emb)
        print(
            f"[Group {cluster_data['group_id']}] "
            f"#clusters = {cluster_data['n_clusters']}"
    )
    print("===== clustering and server collecting Finished =====")

    # Server-level graph convolution
    updated_cluster_emb, global_item_emb = server_graph.run(cluster_data_list)
    print("===== Server Graph Convolution Finished =====")
    print(f"Total clusters updated: {len(updated_cluster_emb)}")
                                                        # updated_cluster_emb:
                                                            # key   = (group_id, cluster_id)
                                                            # value = Tensor(dim,)
    # 下发在 server 端更新后的 (group,cluster) embedding 到各 group
    group_cluster_emb = defaultdict(dict)
    for (gid, cid), emb in updated_cluster_emb.items():
        group_cluster_emb[gid][cid] = emb
    for cluster_data in cluster_data_list:      # 遍历各 group 的 cluster_data（一对一）
        gid = cluster_data['group_id']
        Recmodel = group_models[gid]
        user_emb, _ = Recmodel.computer()
        user_emb = user_emb.clone()
        # 更新
        cluster_users = cluster_data['cluster_users']   # group 内：cluster -> users 映射
        cluster_updated_embs = group_cluster_emb[gid]   # 该 group 内更新后的 cluster embedding
        for cid, users in cluster_users.items():
            if cid not in cluster_updated_embs:
                continue
            for u in users:
                user_emb[u] = user_emb[u] + alpha * cluster_updated_embs[cid]
                # user_emb[u] = cluster_updated_embs[cid]
        Recmodel.embedding_user.weight.data.copy_(user_emb)     # 把更新后的 user embedding 写回模型
        # Recmodel.embedding_item.weight.data.copy_(global_item_emb)    # 用 server 端训练后的 item embedding 更新本地模型
    # group-level local training
    for gid, Recmodel in group_models.items():
        dataset = group_datasets[gid]
        bpr = group_bpr[gid]
        print(f"[Round {r}] Group {gid} local training")
        for epoch in range(local_epochs):
            Procedure.BPR_train_original(
                dataset,
                Recmodel,
                bpr,
                epoch,
                neg_k=Neg_k,
                w=w
            )
    print(f"----------- [ROUND {r} TEST] ----------")
    for gid, Recmodel in group_models.items():
        dataset = group_datasets[gid]
        print(f"[ROUND {r} TEST] : Group {gid}")
        Procedure.Test(
            dataset,
            Recmodel,
            0,
            w,
            world.config['multicore'],
            test=1
        )
print("\n================ All Global Rounds Finished ================")

print("\n================ FINAL TEST ================")
for gid, Recmodel in group_models.items():
    dataset = group_datasets[gid]
    print(f"[FINAL TEST] Group {gid}")
    Procedure.Test(
        dataset,
        Recmodel,
        0,
        w,
        world.config['multicore'],
        test=1
    )