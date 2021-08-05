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
    # df_click['vts_ratio'].isnull().sum()

    df_click.sort_values(by = 'time', inplace = True)
    # df_click = df_click.drop_duplicates(['did', 'fvid', 'vid', 'vts_ratio']).reset_index(drop=True)
    
    ### show ###    
    # df_show = pd.read_feather(data_path + 'dbfeed_show_info.feather')

    # 计算日期 2021-03-20 ~ 2021-03-26
    df_click['date'] = df_click['time'].apply(timestamp_to_date)
    # df_show['date'] = df_show['time'].apply(timestamp_to_date)
    # tdate = df_show['date'].unique()
    
    # temp = df_show.head(1000)
    # temp = df_show.groupby(['date', 'did', 'fvid'])['vid'].count().reset_index()
    
    # t1 = df_show[df_show['did'] == '182d818a8d95b642ebd6424ed907c0b9'][df_show['fvid'] == 4483065]
    # t2 = df_click[df_click['did'] == '46c0551471b8693aec72dd2a5df6f171'][df_click['fvid'] == 3250366]
    
    # train 2021-03-25
    train_click = df_click[df_click['date'] == '2021-03-25']
    # train_click['vid'].nunique() # fvid 7266  vid 14231
    # train_click['label'] = 1
    # train_click = train_click.sort_values('time').reset_index(drop=True)        
    train_candidate_did_fvid = train_click[['did', 'fvid']].drop_duplicates().reset_index(drop=True)
    train_candidate_vid = train_click[['vid']].drop_duplicates().reset_index(drop=True)
    
    train_label = train_click.groupby(['did', 'fvid', 'vid'])['vts_ratio'].sum().reset_index()
    train_label['vts_ratio'] = train_label['vts_ratio'].apply(
                               lambda x : 1 if x >= 1 else x).apply(lambda x : 0 if x <= 0 else x)

    # recall
    df_recall = df_click[df_click['date'] < '2021-03-25']
    df_recall = df_recall[df_recall['vid'].isin(train_candidate_vid['vid'])]
   
    df_recall = df_recall.groupby(['fvid', 'vid'])['vts_ratio'].sum().reset_index()
    # df_recall['vts_ratio'] = df_recall['vts_ratio'].apply(lambda x : 1 if x >= 1 else x).apply(lambda x : 0 if x <= 0 else x)
    # df_recall = df_recall.groupby(['fvid', 'vid'])['vts_ratio'].mean().reset_index()     
    # df_recall['n'] = df_recall.groupby('fvid')['vts_ratio'].rank()
    # df_recall['m'] = df_recall.groupby('fvid')['n'].transform(max)
    # df_recall['vts_ratio'] = df_recall['vts_ratio'] / df_recall['vts_ratio'].max()
    # df_recall['s'] = df_recall['n'] / df_recall['m']
    
    reader = Reader(rating_scale = (0, 1))
    trainset = df_recall[['fvid', 'vid', 'vts_ratio']].drop_duplicates()
    trainset = Dataset.load_from_df(trainset, reader)
    trainset = trainset.build_full_trainset()

    # algo = SVD()
    sim_options = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options=sim_options)
    # algo = KNNBaseline()    
    algo.fit(trainset)
    
    testset_fvid = train_candidate_did_fvid[train_candidate_did_fvid['fvid'].isin(df_recall['fvid'])]['fvid'].unique()
    # testset_fvid = testset_fvid[:500]
    testset_vid = train_candidate_vid[train_candidate_vid['vid'].isin(df_recall['vid'])]['vid'].unique()
    testset = [(x, y, 0) for x in testset_fvid for y in testset_vid]
    
    # train_recall = []
    # for did in tqdm(testset_did):
    #     # pass
    #     testset = [(did, x, 0) for x in testset_vid]
    predictions = algo.test(testset)
    
    train_recall = get_top_n(predictions, n = 100)
    # train_recall = train_recall.rename(columns = {'did':'fvid'}).reset_index(drop = True)
    # train_recall.append(temp_recall)
    # train_recall = pd.concat(train_recall)    
    # train_recall =  get_top_n(predictions, n = 100)
    
    train_recall = pd.merge(train_candidate_did_fvid, train_recall, on = ['fvid'], how = 'left')
    train_recall = pd.merge(train_recall, train_label, on = ['did', 'fvid', 'vid'], how = 'left')
    train_recall.loc[~train_recall['vts_ratio'].isnull(), 'label'] = 1
    train_recall['label'].fillna(0, inplace = True)
    train_recall = train_recall.sort_values(by = 'pred', ascending = False).reset_index(drop = True)
    
    train_recall.to_feather(data_path + 'recall/svd_train.feather')    
    
    train_score = evaluate(train_recall)
    print('\ntotal time:', datetime.datetime.now() - start)
