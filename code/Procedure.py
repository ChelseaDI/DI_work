'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score


CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None, isServer=False):
    Recmodel = recommend_model
    Recmodel.train()        # 把模型切换成训练模式
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    if isServer:
        total_batch = len(users) // world.config['server_bpr_batch_size'] + 1
        print(f"Server BPR Training: total_batch = {total_batch}, bpr_batch_size = {world.config['server_bpr_batch_size']}")
    else:
        total_batch = len(users) // world.config['bpr_batch_size'] + 1
        print(f"Client BPR Training: total_batch = {total_batch}, bpr_batch_size = {world.config['bpr_batch_size']}")
    # total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)       # 计算loss，计算梯度，更新参数
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0, test=1, isServer=False, rating_initial=None, item_mask=None):
    if isServer:
        u_batch_size = world.config['server_test_u_batch_size']
        print(f"Server Test: u_batch_size = {u_batch_size}")
    else:
        u_batch_size = world.config['test_u_batch_size']
        print(f"Client Test: u_batch_size = {u_batch_size}")
    # u_batch_size = world.config['test_u_batch_size']

    dataset: utils.BasicDataset
    if test==1:
        testDict: dict = dataset.testDict
    else:
        testDict: dict = dataset.valDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()

    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10      # 至少分十组
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        # total_batch = len(users) // u_batch_size + 1
        total_batch = (len(users) // u_batch_size
               if len(users) % u_batch_size == 0
               else len(users) // u_batch_size + 1)
        
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            if test==1:
                allPos = dataset.getUserPosItems_Test(batch_users)
            else:
                allPos = dataset.getUserPosItems(batch_users)
            # allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()      # 将用户ID移到GPU
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)       # 获取用户评分预测
            if rating_initial is not None and not isServer:     # 使用所属 cluster 的评分进行微调
                rating += rating_initial[batch_users_gpu]   
            if item_mask is not None:           # 局限于group内物品
                rating[:, ~item_mask] = -(1 << 10)

            #rating = rating.cpu()
            # 排除已交互物品
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)         # 将已交互物品的评分置为负无穷
            _, rating_K = torch.topk(rating, k=max_K)               # 获取Top-K推荐

            # 数据转移和清理
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        print(f"Debug info:")
        print(f"  len(users) = {len(users)}")
        print(f"  u_batch_size = {u_batch_size}")
        print(f"  total_batch (calculated) = {total_batch}")
        print(f"  len(users_list) (actual) = {len(users_list)}")
        print(f"  len(users) % u_batch_size = {len(users) % u_batch_size}")
        assert total_batch == len(users_list)           # 这代码不就纯纯有毛病么，必须得有余数才行
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        # return results
        
        if test==0:
            return results['recall'][0]
        else:
            return results

    
def server_test(server_recmodel):
    """
    返回：
        server_rating_cluster : Tensor
            shape = (n_clusters_all, n_items)
    """
    server_recmodel.eval()
    with torch.no_grad():
        users_emb, items_emb = server_recmodel.computer()
        rating = server_recmodel.f(torch.matmul(users_emb, items_emb.t()))
    return rating.detach()