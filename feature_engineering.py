# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 23:27:44 2021

@author: tongj
"""


import pandas as pd

import utils

'''
train_recall
train_candidate_did_fvid
train_date
'''

# 天天向上2021属于合集，天天向上属于系列


def get_fvid_vid_same_cid(data, df_vid_info):
    data.loc[data['cid'] == data['fvid_cid'], 'fvid_vid_same_cid'] = 1
    return data


def get_update_vid(data, df_vid_info):
    data.loc[(data['cid'] == data['fvid_cid']) &
             (data['serialno'] < data['fvid_serialno']),
             'update_vid'] = 1
    return data


def get_vid_last_click_time_diff(data, df_click_data, df_vid_info):
    df = data.copy()
    vid_last_click = df_click_data.groupby('vid')['time'].max(
    ).reset_index().rename(columns={'time': 'last_click'})
    vid_last_click = pd.merge(
        vid_last_click, df_vid_info[['vid', 'online_time']])
    vid_last_click['vid_last_click_time_diff'] = vid_last_click['last_click'] - \
        vid_last_click['online_time']
    df = pd.merge(
        df, vid_last_click[['vid', 'vid_last_click_time_diff']], on='vid', how='left')
    return df


def get_vid_online_day_diff(data, end_date):
    df = data.copy()
    df['date'] = end_date
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['online_date'] = pd.to_datetime(df['online_date'], format='%Y-%m-%d')
    df['vid_online_day_diff'] = df['date'] - df['online_date']
    df['vid_online_day_diff'] = df['vid_online_day_diff'].dt.days
    df.drop('date', axis=1, inplace=True)
    return df


def feature_click_unique(data, df_click_data, group_key, key):
    df_feats = df_click_data.groupby(group_key)[key].nunique()
    data[f'nunique_click_{group_key}_{key}'] = data[group_key].map(df_feats)
    return data


def feature_show_unique(data, df_show_data, group_key, key):
    df_feats = df_show_data.groupby(group_key)[key].nunique()
    data[f'nunique_show_{group_key}_{key}'] = data[group_key].map(df_feats)
    return data


def feature_cross_count(data, df, group_list):
    df_feats = df.groupby(group_list)['time'].count().reset_index().rename(
        columns={'time': f'cross_count_{group_list[0]}_{group_list[1]}'})
    data = data.merge(df_feats, on=group_list, how='left')
    return data


def feature_cross_sum(data, df_click_data, group_list):
    df_feats = df_click_data.groupby(group_list)['vts_ratio'].sum().reset_index().rename(
        columns={'vts_ratio': f'cross_sum_{group_list[0]}_{group_list[1]}'})
    data = data.merge(df_feats, on=group_list, how='left')
    return data


def feature_cross_mean(data, df_click_data, group_list):
    df_feats = df_click_data.groupby(group_list)['vts_ratio'].mean().reset_index().rename(
        columns={'vts_ratio': f'cross_mean_{group_list[0]}_{group_list[1]}'})
    data = data.merge(df_feats, on=group_list, how='left')
    return data


def make_features(df_recall, df_click, df_show, df_vid_info, candidate_did_fvid, end_date):
    '''
    train_features = make_features(
        train_recall, df_click, df_show, df_vid_info, train_candidate_did_fvid, train_date)

    df_recall = train_recall.copy()
    candidate_did_fvid = train_candidate_did_fvid.copy()
    end_date = train_date
    '''
    data = df_recall.copy()
    data = data.merge(df_vid_info, on='vid', how='left')
    df_fvid = df_vid_info[['vid', 'cid', 'serialno']].rename(
        columns={'vid': 'fvid', 'cid': 'fvid_cid', 'serialno': 'fvid_serialno'})
    data = pd.merge(data, df_fvid, on='fvid', how='left')

    df_click_data = df_click[df_click['date'] < end_date]
    df_show_data = df_show[df_show['date'] < end_date]

    data = get_fvid_vid_same_cid(data, df_vid_info)
    data = get_update_vid(data, df_vid_info)

    # vid最近一次点击时间差
    data = get_vid_last_click_time_diff(data, df_click_data, df_vid_info)

    # vid上线天数
    data = get_vid_online_day_diff(data, end_date)

    # df_click
    for i in ['fvid', 'vid', 'cid']:
        data = feature_click_unique(data, df_click_data, 'did', i)
        data = feature_click_unique(data, df_click_data, i, 'did')

    # df_show
    for i in ['fvid', 'vid', 'cid']:
        data = feature_show_unique(data, df_show_data, 'did', i)
        data = feature_show_unique(data, df_show_data, i, 'did')

    # cross count
    for i in ['fvid', 'vid', 'cid']:
        data = feature_cross_count(data, df_click_data, ['did', i])

    for i in ['vid', 'cid']:
        data = feature_cross_count(data, df_click_data, ['fvid', i])

    for i in ['fvid', 'vid', 'cid']:
        data = feature_cross_sum(data, df_click_data, ['did', i])
        data = feature_cross_mean(data, df_click_data, ['did', i])

    for i in ['vid', 'cid']:
        data = feature_cross_sum(data, df_click_data, ['fvid', i])
        data = feature_cross_mean(data, df_click_data, ['fvid', i])

    # 当天用户观看候选合集的次数
    df_current_did_cid_clicks = candidate_did_fvid.groupby(['did', 'cid'])['fvid'].count(
    ).reset_index().rename(columns={'fvid': 'current_did_cid_clicks'})
    data = data.merge(df_current_did_cid_clicks, on=['did', 'cid'], how='left')
    data.drop('cid', axis=1, inplace=True)

    data.fillna(0, inplace=True)
    return data
