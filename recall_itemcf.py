# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:03:17 2021

@author: tongj
"""

import os
import math
import pickle
import random
import signal
import datetime
import numpy as np
import pandas as pd
import multiprocessing

from surprise import Dataset
from surprise import Reader 
from surprise import SVD
from surprise import KNNBaseline
from surprise import accuracy
from tqdm import tqdm
from random import shuffle
from collections import defaultdict
from joblib import Parallel, delayed

from utils import data_path
from utils import evaluate
from utils import timestamp_to_date

random.seed(25)


def cal_sim(df):
    '''
    df = df_recall.copy()
    df = train_behavior[:10000]
    '''
    user_item_ = df.groupby('fvid')['vid'].agg(lambda x: list(x)).reset_index()
    user_item_['vid'] = user_item_['vid'].apply(lambda x: x[:200] if len(x) > 200 else x)
    # user_item_['n'] = user_item_['vid'].apply(lambda x:  len(x))

    user_item_dict = dict(zip(user_item_['fvid'], user_item_['vid']))
    
    item_cnt = df.groupby('vid')['fvid'].count().reset_index()
    item_cnt = dict(zip(item_cnt['vid'], item_cnt['fvid']))

    sim_dict = {}
    for _, items in tqdm(user_item_dict.items()):
        # pass
        for loc1, item in enumerate(items):
            sim_dict.setdefault(item, {})

            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue

                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重
                # 考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(loc2 - loc1) - 1))

                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    for item, relate_items in tqdm(sim_dict.items()):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])

    return sim_dict, user_item_dict


def recall(candidate_fvid, sim_dict, user_item_dict):
    '''
    candidate_fvid = train_candidate_fvid.copy()
    user_id = 2965081
    '''
    data_list = []

    for user_id in tqdm(candidate_fvid):
        rank = {}

        if user_id not in user_item_dict:
            continue

        interacted_items = user_item_dict[user_id]

        for loc, item in enumerate(interacted_items):
            for relate_item, wij in sorted(sim_dict[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (0.7**loc)

        sim_items = sorted(rank.items(), key=lambda d: d[1],
                           reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['vid'] = item_ids
        df_temp['pred'] = item_sim_scores
        df_temp['fvid'] = user_id
        df_temp = df_temp[['fvid', 'vid', 'pred']]
        data_list.append(df_temp)

    df_data = pd.concat(data_list, sort=False)
    df_data.sort_values(by = 'pred', ascending = False, inplace = True)
    df_data.drop_duplicates(inplace = True)
    return df_data


def get_top_n(predictions, n = 100):
    """
    Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    n = 30
    """


    # df = []
    # for uid, iid, true_r, est, _ in predictions:
    #     df.append([uid, iid, est])
    # df = pd.DataFrame(predictions, columns = ['fvid', 'vid', 'pred']) 
    df = pd.DataFrame(predictions, columns=['fvid', 'vid', 'rui', 'pred', 'details'])  
    df = df[['fvid', 'vid', 'pred']]
    df['pred_rank'] = df.groupby('fvid')['pred'].rank(ascending = False)
    df = df[df['pred_rank'] <= 100]
    return df

def main():

    start = datetime.datetime.now()
    
    ### click ###
    df_click = pd.read_feather(data_path + 'dbfeed_click_info.feather')
    df_click.sort_values(by = 'time', inplace = True)

    # 计算日期 2021-03-20 ~ 2021-03-26
    df_click['date'] = df_click['time'].apply(timestamp_to_date)

    # train 2021-03-25
    train_click = df_click[df_click['date'] == '2021-03-25']
    
    train_candidate_did_fvid = train_click[['did', 'fvid']].drop_duplicates().reset_index(drop=True)
    train_candidate_vid = train_click[['vid']].drop_duplicates().reset_index(drop=True)

    train_candidate_fvid = train_candidate_did_fvid['fvid'].unique()
 
    train_label = train_click.groupby(['did', 'fvid', 'vid'])['vts_ratio'].sum().reset_index()
    train_label['vts_ratio'] = train_label['vts_ratio'].apply(lambda x : 1 if x >= 1 else x).apply(lambda x : 0 if x <= 0 else x)

    # recall
    df_recall = df_click[df_click['date'] < '2021-03-25']
    df_recall = df_recall[df_recall['vid'].isin(train_candidate_vid['vid'])]
    
    df_recall = df_recall.groupby(['fvid', 'vid'])['vts_ratio'].sum().reset_index()
    df_recall = df_recall.sort_values(['fvid', 'vts_ratio'], ascending=[True, False]).reset_index(drop=True)    
    
    sim_dict, user_item_dict = cal_sim(df_recall)    
    
    sim_pkl_file = data_path + 'train_itemcf_sim.pkl'    
    f = open(sim_pkl_file, 'wb')
    pickle.dump(sim_dict, f)
    f.close()

    # 召回
    train_recall = recall(train_candidate_fvid, sim_dict, user_item_dict)
    train_recall = pd.merge(train_candidate_did_fvid, train_recall, on = ['fvid'], how = 'left')
    train_recall = pd.merge(train_recall, train_label, on = ['did', 'fvid', 'vid'], how = 'left')
    train_recall.loc[~train_recall['vts_ratio'].isnull(), 'label'] = 1
    train_recall['label'].fillna(0, inplace = True)
    train_recall = train_recall.sort_values(['fvid', 'pred'], ascending=[True, False]).reset_index(drop=True)
    train_recall.to_pickle(data_path + 'train_recall.pkl')
    
    train_recall.to_feather(data_path + 'recall/itemcf_train.feather')    
    
    train_score = evaluate(train_recall)
    print('\ntotal time:', datetime.datetime.now() - start)
