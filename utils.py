# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 18:04:57 2021

@author: tongj
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data_path = "D:/data/"


def timestamp_to_date(timestamp):
    '''
    timestamp = 1616236257
    '''
    # 获得当前时间时间戳
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    try:
        timeArray = time.localtime(int(timestamp))
        otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
        date = otherStyleTime.split(' ')[0]
    except:
        date = np.nan
    return date


def evaluate(data):
    score = []
    df = data.copy()
    for i in [10, 30, 50, 100]:
        temp = df[df['num'] <= i]
        hitrate = temp.groupby('did')['label'].max().reset_index()
        hitrate = hitrate['label'].sum() / hitrate.shape[0]
        score.append(hitrate)

        labelrate = temp['label'].sum() / temp.shape[0]
        score.append(labelrate)

    score = pd.DataFrame([score],
                         columns=['hitrate_10', 'labelrate_10', 'hitrate_30',
                                  'labelrate_30', 'hitrate_50', 'labelrate_50',
                                  'hitrate_100', 'labelrate_100'])
    score = score[['hitrate_10', 'hitrate_30', 'hitrate_50', 'hitrate_100',
                  'labelrate_10', 'labelrate_30', 'labelrate_50', 'labelrate_100']]
    score = score.round(4)
    return score


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    cols_ = [col for col in list(df) if col not in ['cid', 'vid']]
    for col in cols_:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def test_single_feature_importance(data, feature):
    '''
    feature = 'fvid_click_unique_did'
    '''
    df = data.copy()

    train_data = df[[feature]]
    train_target = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        train_data, train_target, test_size=0.2, random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0)

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

    trn_data = lgb.Dataset(X_train, label=y_train.values)
    val_data = lgb.Dataset(X_valid, label=y_valid.values)

    clf = lgb.train(params,
                    trn_data,
                    valid_sets=[trn_data, val_data],
                    num_boost_round=10000,
                    verbose_eval=100,
                    early_stopping_rounds=100)  # , feval=self_gauc)

    # 验证集的分数
    print('valid', roc_auc_score(y_valid, clf.predict(X_valid)))
    print('test', roc_auc_score(y_test, clf.predict(X_test)))


def get_feature_importance(clf, features):
    feature_imp = pd.DataFrame(
        {'feature': features, 'importance': clf.feature_importance()})
    feature_imp = feature_imp.sort_values(
        by='importance', ascending=False).reset_index(drop=True)
    print(feature_imp.head(10))
    return feature_imp


def AP_N(actual_vids, pred_vids, N):
    if len(pred_vids) > N:
        return 0
    down = np.min([len(actual_vids), N])
    actual_vids = set(actual_vids)
    up, flag, correct = 0, 0, 0
    for pv in pred_vids:
        if flag > N:
            break
        flag += 1
        if pv in actual_vids:
            correct += 1
            up += float(correct) / flag

    return float(up) / down


# MAP@30评分
def cal_map(df_answer, df_solution):
    df_A_map_6 = df_answer.groupby(['did', 'fvid'])[
        'vid'].apply(list).reset_index()
    test_solution_map_6 = df_solution.groupby(
        ['did', 'fvid'])['vid'].apply(list).reset_index()
    test_solution_map_6.rename(columns={'vid': 'pred_vid'}, inplace=True)
    df_A_map_6 = df_A_map_6.merge(test_solution_map_6, on=[
                                  'did', 'fvid'], how='left')
    return df_A_map_6.apply(lambda x: AP_N(x["vid"], x["pred_vid"], 30), axis=1).mean()


# Task2评分
def cal_task2_score(df_answer, df_solution):
    test_t2 = df_answer.merge(
        df_solution, on=['did', 'fvid', 'vid'], how='left')
    test_t2 = test_t2.rename(columns={'vts_ratio_x': 'actual_vts_ratio'}).rename(
        columns={'vts_ratio_y': 'pred_vts_ratio'})
    df_score = test_t2[test_t2['pred_vts_ratio'].notnull()
                       ].reset_index(drop=True)
    df_score['T2_Score'] = 1 / \
        (1 +
         np.sqrt(np.abs(df_score['actual_vts_ratio'] - df_score['pred_vts_ratio'])))
    df_score = df_score.groupby(['did', 'fvid'])[
        'T2_Score'].sum().reset_index()
    df_temp = df_answer.groupby(['did', 'fvid'])['vid'].count().reset_index()
    df_score = df_temp.merge(df_score, on=['did', 'fvid'], how='left')
    df_score['T2_Score'].fillna(0, inplace=True)
    df_score['T2_Score'] = df_score['T2_Score'] / df_score['vid']
    S = len(df_answer.drop_duplicates(['did', 'fvid']))
    t2_score = float(df_score['T2_Score'].sum()) / float(S)
    return t2_score
