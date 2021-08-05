# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:20:40 2021

@author: tongj
"""

import utils
import recall_ruler
from feature_engineering import make_features

import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def save_feather(data_path):

    df_click_set = []
    df_show_set = []
    df_behavior_set = []

    for part in tqdm(['part_1', 'part_2', 'part_3', 'part_4', 'part_5', 'part_6',
                      'part_7', 'part_8']):
        # continue

        df_click = pd.read_csv(
            utils.data_path + "{}/dbfeed_click_info.csv".format(part))

        df_show = pd.read_csv(
            utils.data_path + "{}/dbfeed_show_info.csv".format(part))

        # df_behavior = pd.read_csv(
        #     utils.data_path + "{}/user_main_behavior.csv".format(part))

        # did = df_click['did'].unique()
        # fvid = df_click['fvid'].unique()

        # df_show = df_show[df_show['did'].isin(did)]
        # df_show = df_show[df_show['fvid'].isin(fvid)]
        # df_behavior = df_behavior[df_behavior['did'].isin(did)]

        df_click['date'] = df_click['time'].apply(utils.timestamp_to_date)
        df_show['date'] = df_show['time'].apply(utils.timestamp_to_date)
        # df_behavior['date'] = df_behavior['timestamp'].apply(
        #     utils.timestamp_to_date)

        # tdate = np.sort(df_behavior['date'].unique())[-30]
        # df_behavior = df_behavior[df_behavior['date'] >= tdate]

        df_click_set.append(df_click)
        df_show_set.append(df_show)
        # df_behavior_set.append(df_behavior)

    df_click = pd.concat(df_click_set).reset_index(drop=True)
    df_show = pd.concat(df_show_set).reset_index(drop=True)
    # df_behavior = pd.concat(df_behavior_set).reset_index(drop=True)

    df_click.to_feather(utils.data_path + 'dbfeed_click_info.feather')
    df_show.to_feather(utils.data_path + 'dbfeed_show_info.feather')
    # df_behavior.to_feather(utils.data_path + 'user_main_behavior.feather')


# 候选的did-fvid
def get_candidate_did_fvid(df_click, df_vid_info):
    candidate_did_fvid = df_click[['did', 'fvid']
                                  ].drop_duplicates().reset_index(drop=True)
    candidate_did_fvid = candidate_did_fvid.merge(
        df_vid_info[['vid', 'cid']].rename(columns={'vid': 'fvid'}),
        on='fvid', how='left')
    return candidate_did_fvid


# 候选的vid
def get_candidate_vid(df_click, df_vid_info):
    candidate_vid = df_click[['vid']].drop_duplicates().reset_index(drop=True)
    candidate_vid = candidate_vid.merge(df_vid_info[['vid', 'cid', 'online_time']],
                                        on='vid', how='left')
    return candidate_vid


def recall(df_click, df_show, tdate, candidate_did_fvid, candidate_vid, top_n):
    '''
    tdate = valid_date
    candidate_did_fvid = valid_candidate_did_fvid.copy()
    candidate_vid = valid_candidate_vid.copy()
    top_n = 60
    '''

    df_fvid_topN_vts_show = recall_ruler.recall_by_fvid_topN_vts_show(
        df_show, tdate, candidate_did_fvid, candidate_vid, 30)

    # 过N天fvid下，vid的点击率，观看比例的TOPN进行召回
    df_fvid_topN_vts_ratio = recall_ruler.recall_by_fvid_topN_vts_ratio(
        df_click, tdate, candidate_did_fvid, candidate_vid, 20)

    # 过N天cid下，vid的观看比例的TOPN进行召回
    df_vid_hot_cid = recall_ruler.recall_by_hot_cid(
        df_click, tdate, candidate_did_fvid, candidate_vid, 10)

    # 过N天cid下，vid的最新上线日期的TOPN进行召回
    df_vid_new_cid = recall_ruler.recall_by_new_cid(
        df_click, tdate, candidate_did_fvid, candidate_vid, 10)

    # 过N天的热门(vid的点击次数，观看比例的TOPN进行召回)
    df_hot_vid_list = recall_ruler.recall_by_hot_data(
        df_click, tdate, candidate_did_fvid, candidate_vid, 60)

    df_recall = candidate_did_fvid.merge(
        df_fvid_topN_vts_show, on='fvid', how='left')

    df_recall = df_recall.merge(
        df_fvid_topN_vts_ratio, on='fvid', how='left')

    df_recall = df_recall.merge(
        df_vid_hot_cid, on='cid', how='left')

    df_recall = df_recall.merge(
        df_vid_new_cid, on='cid', how='left')

    df_recall = df_recall.merge(
        df_hot_vid_list, on=['did', 'fvid', 'cid'], how='left')

    df_recall['vid_show'] = df_recall['vid_show'].fillna(
        0).apply(lambda x: [] if x == 0 else x)
    df_recall['vts_ratio'] = df_recall['vts_ratio'].fillna(
        0).apply(lambda x: [] if x == 0 else x)
    df_recall['hot_cid'] = df_recall['hot_cid'].fillna(
        0).apply(lambda x: [] if x == 0 else x)
    df_recall['new_cid'] = df_recall['new_cid'].fillna(
        0).apply(lambda x: [] if x == 0 else x)

    df_recall['recall_list'] = (df_recall['vid_show'] + df_recall['vts_ratio'] + df_recall['hot_vid'] +
                                df_recall['hot_cid'] + df_recall['new_cid'])

    df_recall = df_recall[['did', 'fvid', 'recall_list']]
    data = df_recall.explode('recall_list').rename(
        columns={'recall_list': 'vid'})
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
    # recall_score = utils.evaluate(data)

    return data


def main():

    start = datetime.datetime.now()

    offline = True

    if offline:
        train_date = '2021-03-25'
        valid_date = '2021-03-26'
        test_date = '2021-03-27'
    else:
        train_date = '2021-03-26'
        valid_date = '2021-03-27'
        test_date = '2021-03-28'

    # 点播信息流推荐模块点击日志
    df_click = pd.read_feather(utils.data_path + 'dbfeed_click_info.feather')

    # 点播信息流推荐模块曝光日志
    df_show = pd.read_feather(utils.data_path + 'dbfeed_show_info.feather')

    # 读取视频信息表
    df_vid_info = pd.read_csv(utils.data_path + "vid_info.csv")
    df_vid_info['online_date'] = df_vid_info['online_time'].apply(
        utils.timestamp_to_date)

    # 增加合集字段 cid
    df_click = df_click.merge(
        df_vid_info[['vid', 'cid']], on='vid', how='left')
    df_show = df_show.merge(df_vid_info[['vid', 'cid']], on='vid', how='left')

    # 训练集 ###
    # did, fvid候选集、vid候选集
    train_click = df_click[(df_click['date'] == train_date)
                           ].reset_index(drop=True)
    train_candidate_did_fvid = get_candidate_did_fvid(train_click, df_vid_info)
    train_candidate_vid = get_candidate_vid(train_click, df_vid_info)

    # 验证集 ###
    # did, fvid候选集、vid候选集
    valid_click = df_click[(df_click['date'] == valid_date)
                           ].reset_index(drop=True)
    valid_candidate_did_fvid = get_candidate_did_fvid(valid_click, df_vid_info)
    valid_candidate_vid = get_candidate_vid(valid_click, df_vid_info)

    ### 测试集 ###
    if offline:
        test_click = df_click[(df_click['date'] == test_date)
                              ].reset_index(drop=True)
        test_candidate_did_fvid = get_candidate_did_fvid(
            test_click, df_vid_info)
        test_candidate_vid = get_candidate_vid(test_click, df_vid_info)

    else:
        # did, fvid候选集
        test_candidate_did_fvid = pd.read_csv(
            utils.data_path + "/part_test/test_candidate_did_fvid.csv")

        # 测试集vid候选集
        test_candidate_vid = pd.read_csv(
            utils.data_path + "/part_test/test_candidate_vid.csv")

        test_candidate_did_fvid = test_candidate_did_fvid.merge(
            df_vid_info[['vid', 'cid']].rename(columns={"vid": "fvid"}), on='fvid', how='left')
        test_candidate_vid = test_candidate_vid.merge(
            df_vid_info[['vid', 'cid', 'online_time']], on='vid', how='left')

    # 召回构造训练样本
    train_recall = recall(df_click, df_show, train_date,
                          train_candidate_did_fvid, train_candidate_vid, 60)
    valid_recall = recall(df_click, df_show, valid_date,
                          valid_candidate_did_fvid, valid_candidate_vid, 60)
    test_recall = recall(df_click, df_show, test_date,
                         test_candidate_did_fvid, test_candidate_vid, 60)

    score_recall_train = utils.evaluate(train_recall)
    score_recall_valid = utils.evaluate(valid_recall)
    score_recall_test = utils.evaluate(test_recall)

    print("提取训练集的特征")
    train_features = make_features(
        train_recall, df_click, df_show, df_vid_info, train_candidate_did_fvid, train_date)

    print("提取验证集的特征")
    valid_features = make_features(
        valid_recall, df_click, df_show, df_vid_info, valid_candidate_did_fvid, valid_date)

    print("提取测试集的特征")
    test_features = make_features(
        test_recall, df_click, df_show, df_vid_info, test_candidate_did_fvid, test_date)

    train_data = utils.reduce_mem_usage(train_features, verbose=False)
    valid_data = utils.reduce_mem_usage(valid_features, verbose=False)
    test_data = utils.reduce_mem_usage(test_features, verbose=False)

    # train_data.drop('key_word', axis=1, inplace=True)
    # valid_data.drop('key_word', axis=1, inplace=True)
    # test_data.drop('key_word', axis=1, inplace=True)

    # train_data.to_feather(utils.data_path + 'train_features.feather')
    # valid_data.to_feather(utils.data_path + 'valid_features.feather')
    # test_data.to_feather(utils.data_path + 'test_features.feather')

    # train_data = pd.read_feather(
    #     utils.data_path + 'train_features.feather')
    # valid_data = pd.read_feather(
    #     utils.data_path + 'valid_features.feather')
    # test_data = pd.read_feather(
    #     utils.data_path + 'test_features.feather')

    # ['did', 'fvid', 'vid', 'num', 'label', 'is_intact', 'online_time',
    #  'serialno', 'series_id', 'duration', 'title_length', 'key_word',
    #  'online_date', 'fvid_cid', 'fvid_serialno', 'fvid_vid_same_cid',
    #  'vid_last_click_time_diff', 'vid_online_day_diff',
    #  'nunique_click_did_fvid', 'nunique_click_fvid_did',
    #  'nunique_click_did_vid', 'nunique_click_vid_did',
    #  'nunique_click_did_cid', 'nunique_click_cid_did',
    #  'nunique_show_did_fvid', 'nunique_show_fvid_did',
    #  'nunique_show_did_vid', 'nunique_show_vid_did', 'nunique_show_did_cid',
    #  'nunique_show_cid_did', 'cross_count_did_fvid', 'cross_count_did_vid',
    #  'cross_count_did_cid', 'cross_count_fvid_vid', 'cross_count_fvid_cid',
    #  'cross_sum_did_fvid', 'cross_mean_did_fvid', 'cross_sum_did_vid',
    #  'cross_mean_did_vid', 'cross_sum_did_cid', 'cross_mean_did_cid',
    #  'cross_sum_fvid_vid', 'cross_mean_fvid_vid', 'cross_sum_fvid_cid',
    #  'cross_mean_fvid_cid', 'current_did_cid_clicks']

    useless_cols = ['did', 'fvid', 'vid', 'label', 'online_time', 'preds', 'preds_rank',
                    'vts_ratio', 'series_id', 'key_word', 'online_date', 'fvid_cid']

    ### 用train训练预测test ###
    features = train_data.columns[~train_data.columns.isin(
        useless_cols)].values
    print(features)

    category_features = ['is_intact']
    category_features = [col for col in category_features if col in features]
    for col in category_features:
        train_data[col] = train_data[col].astype('category')
        valid_data[col] = valid_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

    params = {
        'objective': 'binary',  # 定义的目标函数
        'metric': 'auc',
        'boosting_type': 'gbdt',

        'learning_rate': 0.05,
        'max_depth': 12,
        'num_leaves': 2 ** 6,

        'min_child_weight': 10,
        'min_data_in_leaf': 40,

        'feature_fraction': 0.70,
        'subsample': 0.75,
        'seed': 25,

        'nthread': -1,
        'bagging_freq': 1,
        'verbose': -1,
        # 'scale_pos_weight':200
    }

    # useless_did = train_data.groupby('did')['label'].max().reset_index()
    # useless_did = useless_did[useless_did['label'] == 0]['did']
    # train_data = train_data[~train_data['did'].isin(useless_did)]

    trn_data = lgb.Dataset(
        train_data[features], label=train_data['label'].values)
    val_data = lgb.Dataset(
        valid_data[features], label=valid_data['label'].values)

    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    num_boost_round=10000,
                    verbose_eval=100,
                    early_stopping_rounds=100)  # , feval=self_gauc)
    feature_imp = utils.get_feature_importance(clf, features)

    # 验证集的分数
    valid_data['preds'] = clf.predict(
        valid_data[features], num_iteration=clf.best_iteration)

    valid_data = valid_data.sort_values(
        by=['did', 'fvid', 'preds'], ascending=False).reset_index(drop=True)
    valid_data['preds_rank'] = valid_data.groupby(
        ['did', 'fvid'])['vid'].cumcount() + 1

    valid_solution = valid_data[valid_data['preds_rank'] <= 30][[
        'did', 'fvid', 'vid']]
    valid_solution['vts_ratio'] = 1
    valid_solution = valid_solution.drop_duplicates(
        ['did', 'fvid', 'vid']).reset_index(drop=True)

    df_valid_answer = valid_click.groupby(['did', 'fvid', 'vid'])[
        'vts_ratio'].sum().reset_index()
    df_valid_answer['vts_ratio'] = df_valid_answer['vts_ratio'].apply(
        lambda x: 1 if x >= 1 else x).apply(lambda x: 0 if x <= 0 else x)
    map_6 = utils.cal_map(df_valid_answer, valid_solution)
    task_2 = utils.cal_task2_score(df_valid_answer, valid_solution)
    print('valid score:', map_6 * 0.7 + task_2 * 0.3, '\n')

    # valid score: 0.2145566386430361
    # valid score: 0.19014469700376652

    # 测试集的分数
    if offline:
        test_data['preds'] = clf.predict(
            test_data[features], num_iteration=clf.best_iteration)
        test_data = test_data.sort_values(
            by=['did', 'fvid', 'preds'], ascending=False).reset_index(drop=True)
        test_data['preds_rank'] = test_data.groupby(
            ['did', 'fvid'])['vid'].cumcount() + 1

        test_solution = test_data[test_data['preds_rank']
                                  <= 30][['did', 'fvid', 'vid']]
        test_solution['vts_ratio'] = 1  # 0.1751
        test_solution = test_solution.drop_duplicates(
            ['did', 'fvid', 'vid']).reset_index(drop=True)

        df_test_answer = test_click.groupby(['did', 'fvid', 'vid'])[
            'vts_ratio'].sum().reset_index()
        df_test_answer['vts_ratio'] = df_test_answer['vts_ratio'].apply(
            lambda x: 1 if x >= 1 else x).apply(lambda x: 0 if x <= 0 else x)
        map_6 = utils.cal_map(df_test_answer, test_solution)
        task_2 = utils.cal_task2_score(df_test_answer, test_solution)
        print('test score:', map_6 * 0.7 + task_2 * 0.3, '\n')
        # test score: 0.18362566528172575

    ### 用valid训练预测test ###
    features = valid_data.columns[~valid_data.columns.isin(
        useless_cols)].values
    print(features)

    trn_data = lgb.Dataset(
        valid_data[features], label=valid_data['label'].values)

    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data],
                    num_boost_round=clf.best_iteration,
                    verbose_eval=100,
                    early_stopping_rounds=100)  # , feval=self_gauc)
    feature_imp = utils.get_feature_importance(clf, features)

    test_data['preds'] = clf.predict(
        test_data[features], num_iteration=clf.best_iteration)
    test_data = test_data.sort_values(
        by=['did', 'fvid', 'preds'], ascending=False).reset_index(drop=True)
    test_data['preds_rank'] = test_data.groupby(
        ['did', 'fvid'])['vid'].cumcount() + 1

    test_solution = test_data[test_data['preds_rank']
                              <= 30][['did', 'fvid', 'vid']]
    test_solution['vts_ratio'] = 1  # 0.1751
    test_solution = test_solution.drop_duplicates(
        ['did', 'fvid', 'vid']).reset_index(drop=True)
    test_solution.to_csv('test_solution.csv', index=None)

    if offline:
        df_test_answer = test_click.groupby(['did', 'fvid', 'vid'])[
            'vts_ratio'].sum().reset_index()
        df_test_answer['vts_ratio'] = df_test_answer['vts_ratio'].apply(
            lambda x: 1 if x >= 1 else x).apply(lambda x: 0 if x <= 0 else x)
        map_6 = utils.cal_map(df_test_answer, test_solution)
        task_2 = utils.cal_task2_score(df_test_answer, test_solution)
        print('test score:', map_6 * 0.7 + task_2 * 0.3, '\n')
        # test score: 0.19024047846589837

    print('total time:',  datetime.datetime.now() - start)


'''
if __name__ == '__main__':
    start = datetime.datetime.now()
    main()
    print('total time:',  )
'''
