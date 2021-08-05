# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:11:35 2021

@author: tongj
"""

import numpy as np
import pandas as pd


# 过去N天fvid下的topN观看比例进行召回
def recall_by_fvid_topN_vts_ratio(df_click, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = test_date
    candidate_did_fvid = test_candidate_did_fvid.copy()
    candidate_vid = test_candidate_vid.copy()
    top_n = 100
    '''
    past_click = df_click[df_click['date'] < tdate]
    # start_date = np.sort(past_click['date'].unique())[-7]
    # past_click = past_click[past_click['date'] >= start_date]

    past_click = past_click[past_click['fvid'].isin(
        candidate_did_fvid['fvid'])]
    past_click = past_click[past_click['vid'].isin(candidate_vid['vid'])]

    df_fvid_vid_ratio = past_click.groupby(['fvid', 'vid'])['vts_ratio'].sum(
    ).reset_index().rename(columns={'vts_ratio': 'vid_sum_vts_ratio'})
    df_fvid_vid_ratio['n'] = df_fvid_vid_ratio.groupby(
        'fvid')['vid_sum_vts_ratio'].rank(method='min', ascending=False)

    # 过去N天，fvid下，观看时长最高的N个视频
    df_recall = df_fvid_vid_ratio[df_fvid_vid_ratio['n'] <= top_n]
    df_recall = df_recall.sort_values(by=['fvid', 'n'])
    df_recall = df_recall.groupby('fvid')['vid'].apply(
        lambda x: list(x)).reset_index().rename(columns={'vid': 'vts_ratio'})
    return df_recall


# 候选视频中历史点击次数最高的N个视频来填补
def recall_by_hot_data(df_click, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = train_date
    candidate_did_fvid = train_candidate_did_fvid.copy()
    candidate_vid = train_candidate_vid.copy()
    top_n = 100
    '''
    past_click = df_click[df_click['date'] < tdate]
    past_click = past_click[past_click['vid'].isin(candidate_vid['vid'])]

    df_top_click_vid = past_click['vid'].value_counts().reset_index().rename(
        columns={'index': 'vid', 'vid': 'click_count'})

    # 取并集
    hot_vid_list = list(df_top_click_vid['vid'].values[:top_n])
    df_recall = candidate_did_fvid.copy()
    df_recall['hot_vid'] = None
    df_recall['hot_vid'] = df_recall['hot_vid'].apply(
        lambda x:  hot_vid_list if x is None else x)
    return df_recall


def recall_by_hot_cid(df_click, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = test_date
    candidate_did_fvid = test_candidate_did_fvid.copy()
    candidate_vid = test_candidate_vid.copy()
    top_n = 100
    '''
    past_click = df_click[df_click['date'] < tdate]
    past_click = past_click[past_click['vid'].isin(candidate_vid['vid'])]
    past_click = past_click[past_click['cid'].isin(candidate_did_fvid['cid'])]

    df_cid_vid_ratio = past_click.groupby(['cid', 'vid'])['vts_ratio'].sum(
    ).reset_index().rename(columns={'vts_ratio': 'vid_sum_vts_ratio'})
    df_cid_vid_ratio['n'] = df_cid_vid_ratio.groupby(
        'cid')['vid_sum_vts_ratio'].rank(method='min', ascending=False)

    # 过去N天，fvid下，观看时长最高的N个视频
    df_recall = df_cid_vid_ratio[df_cid_vid_ratio['n'] <= top_n]
    df_recall = df_recall.sort_values(by=['cid', 'n'])
    df_recall = df_recall.groupby('cid')['vid'].apply(
        lambda x: list(x)).reset_index().rename(columns={'vid': 'hot_cid'})
    return df_recall


def recall_by_new_cid(df_click, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = test_date
    candidate_did_fvid = test_candidate_did_fvid.copy()
    candidate_vid = test_candidate_vid.copy()
    top_n = 100
    '''
    df_cid_vid_new = candidate_vid[candidate_vid['cid'].isin(
        candidate_did_fvid['cid'])]
    df_cid_vid_new['n'] = df_cid_vid_new.groupby(
        'cid')['online_time'].rank(method='min', ascending=False)

    # 过去N天，fvid下，观看时长最高的N个视频
    df_recall = df_cid_vid_new[df_cid_vid_new['n'] <= top_n]
    df_recall = df_recall.sort_values(by=['cid', 'n'])
    df_recall = df_recall.groupby('cid')['vid'].apply(
        lambda x: list(x)).reset_index().rename(columns={'vid': 'new_cid'})
    return df_recall


# 过去N天fvid下的topN观看比例进行召回
def recall_by_fvid_topN_vts_show(df_show, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = test_date
    candidate_did_fvid = test_candidate_did_fvid.copy()
    candidate_vid = test_candidate_vid.copy()
    top_n = 100
    '''

    past_show = df_show[df_show['date'] < tdate]
    start_date = np.sort(past_show['date'].unique())[-3]
    past_show = past_show[past_show['date'] >= start_date]

    past_show = past_show[past_show['fvid'].isin(
        candidate_did_fvid['fvid'])]
    past_show = past_show[past_show['vid'].isin(candidate_vid['vid'])]

    df_fvid_vid_show = past_show.groupby(['fvid', 'vid'])['time'].count(
    ).reset_index().rename(columns={'time': 'show_times'})
    df_fvid_vid_show['n'] = df_fvid_vid_show.groupby(
        'fvid')['show_times'].rank(method='min', ascending=False)

    # 过去N天，fvid下，观看时长最高的N个视频
    df_recall = df_fvid_vid_show[df_fvid_vid_show['n'] <= top_n]
    df_recall = df_recall.sort_values(by=['fvid', 'n'])
    df_recall = df_recall.groupby('fvid')['vid'].apply(
        lambda x: list(x)).reset_index().rename(columns={'vid': 'vid_show'})

    return df_recall


def recall_test(df_click, tdate, candidate_did_fvid, candidate_vid, top_n):

    # data_recall_list = candidate_did_fvid.copy()
    data_recall_list = candidate_did_fvid.merge(
        df_recall, on='fvid', how='left')

    # data_recall_list['recall_list'] = None
    # data_recall_list['recall_list'] = data_recall_list['recall_list'].apply(
    #     lambda x:  today_vid_list if x is None else x)

    data = data_recall_list.explode('vts_ratio')
    data.rename(columns={'vts_ratio': 'vid'}, inplace=True)
    data = data[data['fvid'] != data['vid']]
    data.drop_duplicates(keep='first', inplace=True)
    data['num'] = data.groupby(
        ['did', 'fvid'])['vid'].cumcount() + 1
    data = data[data['num'] <= top_n]

    # label
    df_label = df_click[df_click['date'] == tdate][[
        'did', 'fvid', 'vid']].drop_duplicates()
    df_label['label'] = 1
    data = data.merge(df_label,
                      on=['did', 'fvid', 'vid'], how='left').reset_index(drop=True)
    data['label'].fillna(0, inplace=True)

    recall_score = utils.evaluate(data)
