import world
import dataloader
import model
import utils
from pprint import pprint

import splitter

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
    m_item = dataset.m_item     # 记录全局物品数
    group_size = world.config['group_size']
    seed = world.seed
    sub_datasets = splitter.SubgraphSplitter(group_size, seed).split(dataset)

elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

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

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}