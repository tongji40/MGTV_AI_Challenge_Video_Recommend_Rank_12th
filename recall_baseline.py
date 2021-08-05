# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:53 2021

@author: tongj
"""

import os
import time
import pandas as pd
from utils import data_path
from utils import evaluate
from utils import timestamp_to_date

# 候选的did-fvid
def get_candidate_did_fvid(df_click, df_vid_info):
    candidate_did_fvid = df_click[['did', 'fvid']].drop_duplicates().reset_index(drop=True)
    candidate_did_fvid = candidate_did_fvid.merge(
                         df_vid_info[['vid', 'cid']].rename(columns = {'vid' : 'fvid'}),
                         on='fvid', how='left')
    return candidate_did_fvid

# 候选的vid
def get_candidate_vid(df_click, df_vid_info):
    candidate_vid = df_click[['vid']].drop_duplicates().reset_index(drop=True)
    candidate_vid = candidate_vid.merge(df_vid_info[['vid', 'cid', 'online_time']],
                                        on='vid', how='left')
    return candidate_vid

def isNan_2(a):
    return a != a

def get_candidate_recall(recall_list, candidate_vid_set):
    result = []
    for recall in recall_list:
        if recall in candidate_vid_set:
            result.append(recall)
            
    return result 

#过去N天fvid下的topN点击率进行召回
def recall_by_fvid_topN_ctr(df_click, df_show, topN, recall_name, start_date, end_date):
    df_click = df_click[(df_click['date'] >= start_date) &(df_click['date'] < end_date)]
    df_fvid_vid_clicks = df_click.groupby(['fvid', 'vid'])['pos'].count().reset_index().rename(columns={'pos' : 'click_counts'})
    df_fvid_vid_clicks['click_rank'] = df_fvid_vid_clicks.groupby('fvid')['click_counts'].rank(method='dense', ascending=False)
    
    df_show = df_show[(df_show['date'] >= start_date) &(df_show['date'] < end_date)]
    df_fvid_vid_shows = df_show.groupby(['fvid', 'vid'])['pos'].count().reset_index().rename(columns={'pos' : 'show_counts'})
    df_fvid_vid_clicks = df_fvid_vid_clicks.merge(df_fvid_vid_shows, on=['fvid', 'vid'], how='left')
    df_fvid_vid_clicks['vid_ctr'] = df_fvid_vid_clicks['click_counts'] /(df_fvid_vid_clicks['show_counts'])
    df_fvid_vid_clicks['ctr_rank'] = df_fvid_vid_clicks.groupby('fvid')['vid_ctr'].rank(method='dense', ascending=False)    
    
    #过去N天，fvid下，点击率最高的N个视频
    df_recall = df_fvid_vid_clicks[df_fvid_vid_clicks['ctr_rank'] <= topN]
    df_recall = df_recall.groupby('fvid')['vid'].agg(lambda x : list(x)).reset_index().rename(columns={'vid' : recall_name})
    
    return df_recall

#过去N天fvid下的topN观看比例进行召回
def recall_by_fvid_topN_vts_ratio(df_click, topN, recall_name, start_date, end_date):
    df_click = df_click[(df_click['date'] >= start_date) &(df_click['date'] < end_date)]
    df_fvid_vid_hb_ratio = df_click.groupby(['fvid', 'vid'])['vts_ratio'].sum().reset_index().rename(columns={'vts_ratio' : 'vid_sum_vts_ratio'})
    df_fvid_vid_hb_ratio['vid_sum_vts_ratio_rank'] = df_fvid_vid_hb_ratio.groupby('fvid')['vid_sum_vts_ratio'].rank(method='dense', ascending=False)
         
    #过去N天，fvid下，观看时长最高的N个视频
    df_recall = df_fvid_vid_hb_ratio[df_fvid_vid_hb_ratio['vid_sum_vts_ratio_rank'] <= topN]
    df_recall = df_recall.groupby('fvid')['vid'].agg(lambda x : list(x)).reset_index().rename(columns={'vid' : recall_name})    
    return df_recall

def recall_by_history(df_click_data, df_show_data, candidate_vid, topN, start_date, end_date):
    
    candidate_vid_set = set(candidate_vid['vid'].unique())
    
    # 过去N天fvid下的topN点击率进行召回
    df_recall_1 = recall_by_fvid_topN_ctr(df_click_data, df_show_data, topN, 'topN_fvid_vid_ctr', start_date, end_date)
    
    # 过去N天fvid下的观看比例进行召回
    df_recall_2 = recall_by_fvid_topN_vts_ratio(df_click_data, topN, 'topN_fvid_vid_vts_ratio', start_date, end_date)
    
    # 合并
    df_recall = df_recall_1.merge(df_recall_2, on='fvid', how='outer')
    df_recall['topN_fvid_vid_ctr'] = df_recall['topN_fvid_vid_ctr'].fillna(0).apply(lambda x : [] if x == 0 else x)
    df_recall['topN_fvid_vid_vts_ratio'] = df_recall['topN_fvid_vid_vts_ratio'].fillna(0).apply(lambda x : [] if x == 0 else x)

    # 取并集
    df_recall['recall_list'] =(df_recall['topN_fvid_vid_ctr'] + df_recall['topN_fvid_vid_vts_ratio'] ).apply(set).apply(list)

    # 只选取候选vid中的视频
    df_recall['recall_list'] = df_recall['recall_list'].apply(get_candidate_recall, args=(candidate_vid_set, ))
    
    return df_recall

# 历史N天的热门(点击次数，点击率，观看时长，观看比例)
def recall_by_hot_data(df_click, df_show, candidate_vid, hotN, start_date, end_date):
    '''
    hotN = 40
    '''
    df_click = df_click[(df_click['date'] >= start_date) &(df_click['date'] < end_date)]
    
    # 过去N天，fvid下，vid的点击次数的排序
    df_fvid_vid_clicks = df_click.groupby(['fvid', 'vid'])['pos'].count().reset_index().rename(columns={'pos' : 'click_counts'})
    
    # 过去N天，fvid下，vid的观看比例的排序
    df_fvid_vid_vts_ratio = df_click.groupby(['fvid', 'vid'])['vts_ratio'].sum().reset_index().rename(columns={'vts_ratio' : 'vid_sum_vts_ratio'})

    # 候选视频中历史点击次数最高的N个视频来填补
    df_top_click_vid = df_fvid_vid_clicks.groupby('vid')['click_counts'].sum().reset_index().sort_values('click_counts', ascending=False).reset_index(drop=True)
    df_top_click_vid = df_top_click_vid.merge(candidate_vid, on='vid', how='inner')
    
    # 候选视频中历史曝光观看时长比例最高的N个视频来填补
    df_top_vts_ratio_vid = df_fvid_vid_vts_ratio.groupby('vid')['vid_sum_vts_ratio'].sum().reset_index().sort_values('vid_sum_vts_ratio', ascending=False).reset_index(drop=True)
    df_top_vts_ratio_vid = df_top_vts_ratio_vid.merge(candidate_vid, on='vid', how='inner')
    
    # 取并集
    hot_vid_recall = list(df_top_click_vid['vid'].values[:hotN])+ list(df_top_vts_ratio_vid['vid'].values[:hotN])
    hot_vid_recall = list(set(hot_vid_recall))    
    return hot_vid_recall

def recall_by_hot(train_data, df_click_data, df_show_data, candidate_vid, start_date, end_date, hotN):
    # 增加过去N天的topN热门视频
    hot_vid_recall = recall_by_hot_data(df_click_data, df_show_data, candidate_vid, hotN, start_date, end_date)
    
    # 增加历史热门视频作为召回
    train_data['recall_list'] = train_data['recall_list'].apply(lambda x : list(set(x + hot_vid_recall)))
    return train_data

def explode_df(train_data):
    did_list = []
    fvid_list = []
    vid_list = []
    for row in train_data[['did', 'fvid', 'recall_list']].values :
        did = row[0]
        fvid = row[1]
        recall_list = row[2]
        for recall in recall_list :
            did_list.append(did)
            fvid_list.append(fvid)
            vid_list.append(recall)

    df = pd.DataFrame()
    df['did'] = did_list
    df['fvid'] = fvid_list
    df['vid'] = vid_list
    return df 

def recall(candidate_vid, candidate_did_fvid, df_click, df_show, start_date, end_date, topN = 10):
    '''
    data_train = recall(train_candidate_vid, train_candidate_did_fvid, df_click_data, df_show_data, "2021-03-20", "2021-03-25").drop_duplicates().reset_index(drop=True)
    
    candidate_vid = train_candidate_vid.copy()
    candidate_did_fvid = train_candidate_did_fvid.copy()
    df_click = df_click_data.copy()
    df_show = df_show_data.copy()
    start_date = "2021-03-20"
    end_date = "2021-03-25"
    topN = 10
    
    '''

    # 过N天fvid下，vid的点击率，观看比例的TOPN进行召回
    df_recall = recall_by_history(df_click, df_show, candidate_vid, topN, start_date, end_date)

    data_recall_list = candidate_did_fvid.merge(df_recall, on='fvid', how='left')
    data_recall_list['recall_list'] = data_recall_list['recall_list'].apply(lambda x : [] if isNan_2(x) else x)
    
    # 过N天的热门(vid的点击次数，观看比例的TOPN进行召回)
    data_recall_list = recall_by_hot(data_recall_list, df_click, df_show, candidate_vid, start_date, end_date, 40)

    data = explode_df(data_recall_list)

    # 召回删掉已观看视频
    # df_past = df_click[df_click['date'] < end_date]
    # df_past_click = pd.concat([df_past[['did', 'fvid']].rename(columns = {'fvid':'vid'}),
    #                           df_past[['did', 'vid']]]).drop_duplicates()
    # df_past_click['past_click'] = 1
    # data = pd.merge(data, df_past_click, how = 'left', on = ['did', 'vid'])    
    # data = data[~(data['past_click'] == 1)]
    # data.drop('past_click', axis = 1, inplace = True)
    
    # label 
    df_click['label'] = 1
    data = data.merge(df_click[df_click['date'] == end_date][['did', 'fvid', 'vid', 'label']],
                      on=['did', 'fvid', 'vid'], how='left')
    data['label'].fillna(0, inplace=True)
    return data

# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=5):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)
    
    for k in range(10, topk+1, 10):
        hit_num = 0
        for user, item_list in user_recall_items_dict.items():
            # 获取前k个召回的结果
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if last_click_item_dict[user] in set(tmp_recall_items):
                hit_num += 1
        
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)

def main():

    # 点播信息流推荐模块点击日志 数据重复
    df_click_data = pd.read_feather(data_path + 'dbfeed_click_info.feather')
    # df_click_data.sort_values(by = 'vts_ratio', ascending = False, inplace = True)
    # df_click_data = df_click_data.drop_duplicates(['did', 'fvid', 'vid']).reset_index(drop=True)

    # 点播信息流推荐模块曝光日志
    df_show_data = pd.read_feather(data_path + 'dbfeed_show_info.feather')

    # 读取视频信息表
    df_vid_info = pd.read_csv(data_path + "vid_info.csv")
    #df_vid_info['online_time'] = df_vid_info['online_time'].apply(lambda x : np.NaN if x <= 0 else x)
  
    # 增加合集字段 cid
    df_click_data = df_click_data.merge(df_vid_info[['vid', 'cid']], on='vid', how='left')
    df_show_data = df_show_data.merge(df_vid_info[['vid', 'cid']], on='vid', how='left')
    
    # 计算日期 2021-03-20 ~ 2021-03-26
    df_click_data['date'] = df_click_data['time'].apply(timestamp_to_date)
    df_show_data['date'] = df_show_data['time'].apply(timestamp_to_date)
    
    # 训练集did, fvid候选集  2021-03-25
    train_click = df_click_data[(df_click_data['date'] == '2021-03-25')].reset_index(drop=True)
    train_candidate_did_fvid = get_candidate_did_fvid(train_click, df_vid_info)
    
    # 训练集vid候选集
    train_candidate_vid = get_candidate_vid(train_click, df_vid_info)
    
    # 验证集did, fvid候选集 2021-03-26
    valid_click = df_click_data[(df_click_data['date'] == '2021-03-26')].reset_index(drop=True)
    valid_candidate_did_fvid = get_candidate_did_fvid(valid_click, df_vid_info)
    
    # 验证集vid候选集
    valid_candidate_vid = get_candidate_vid(valid_click, df_vid_info)
    
    # 召回构造训练样本
    data_train = recall(train_candidate_vid, train_candidate_did_fvid, df_click_data, df_show_data, "2021-03-20", "2021-03-25").drop_duplicates().reset_index(drop=True)
    data_valid = recall(valid_candidate_vid, valid_candidate_did_fvid, df_click_data, df_show_data, "2021-03-21", "2021-03-26").drop_duplicates().reset_index(drop=True)

    os.makedirs(data_path + 'recall/', exist_ok=True)
    
    data_train.to_feather(data_path + 'recall/baseline_train.feather')
    data_valid.to_feather(data_path + 'recall/baseline_valid.feather')
    
    train_score = evaluate(data_train)
    valid_score = evaluate(data_valid)


    # Test
    df_click = df_click_data[(df_click_data['date'] >= '2021-03-20') & (df_click_data['date'] < '2021-03-25')]
    temp = train_click[train_click['vid'].isin(df_click['vid'].unique())]
    temp['did'].nunique() / train_candidate_did_fvid['did'].nunique() # 0.9589958511467261

    df_click = df_click_data[(df_click_data['date'] >= '2021-03-21') & (df_click_data['date'] < '2021-03-26')]
    temp = valid_click[valid_click['vid'].isin(df_click['vid'].unique())]
    temp['did'].nunique() / valid_candidate_did_fvid['did'].nunique() # 0.9611330094417454




