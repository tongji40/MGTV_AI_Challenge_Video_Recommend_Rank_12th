# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:10:13 2021

@author: tongj
"""

import time
import pandas as pd
from pandas_profiling import ProfileReport

import utils

data_path = "D:/data/A_CSV/"


def report(data, html_name=''):
    profile = ProfileReport(data,
                            title='Fast Report for EDA',
                            html={'style': {'full_width': True}})

    profile.to_file(f"{html_name}.html")


def eda():

    # ['did', 'fvid', 'vid', 'time',
    #  'pos', 'pageNum', 'aver', 'sver', 'mf',
    #  'mod', 'vts_ratio', 'date']

    # test 当天的vid
    test_candidate_did_fvid = pd.read_csv(
        utils.data_path + "/part_test/test_candidate_did_fvid.csv")

    test_candidate_vid = pd.read_csv(
        utils.data_path + "/part_test/test_candidate_vid.csv")

    df_vid_info = pd.read_csv(utils.data_path + "vid_info.csv")
    df_vid_info['online_date'] = df_vid_info['online_time'].apply(
        utils.timestamp_to_date)
    online_date = df_vid_info[df_vid_info['online_date'] == test_date]

    today_vid = pd.concat([test_candidate_did_fvid[['fvid']].rename(columns={'fvid': 'vid'}),
                          test_candidate_vid[['vid']]]).drop_duplicates()
    today_vid = pd.merge(today_vid, online_date, on='vid')
    temp = today_vid[today_vid['cid'] != 0]
    temp = df_vid_info['cid'].value_counts().reset_index()

    # test 出现的新视频
    # train 0.3282271098306514
    # valid 0.3015024363833243
    # test  0.3114991038908481

    past_vid = df_click[df_click['date'] < valid_date]

    past_vid = pd.concat([past_vid[['fvid']].rename(columns={'fvid': 'vid'}),
                          past_vid[['vid']]]).drop_duplicates()

    temp = valid_candidate_vid[~valid_candidate_vid['vid'].isin(
        past_vid['vid'])]
    len(temp) / len(valid_candidate_vid)

    # train part_6
    train_click = pd.read_csv(data_path + "/part_6/dbfeed_click_info.csv")
    train_did_fvid = train_click[['did', 'fvid']].drop_duplicates()  # 51982

    did_fvid_count = train_did_fvid.groupby(
        'did')['fvid'].count().reset_index()  # did 47483
    did_fvid_count['fvid'].describe()
    len(did_fvid_count[did_fvid_count['fvid'] > 1]) / \
        len(did_fvid_count)  # 0.07109913021502433

    train_candidate_vid = train_click['vid'].unique()  # vid 14231

    # valid part_7
    valid_click = pd.read_csv(data_path + "/part_7/dbfeed_click_info.csv")
    valid_did_fvid = valid_click[['did', 'fvid']].drop_duplicates()  # 59744

    did_fvid_count = valid_did_fvid.groupby(
        'did')['fvid'].count().reset_index()  # did 54545
    did_fvid_count['fvid'].describe()
    len(did_fvid_count[did_fvid_count['fvid'] > 1]) / \
        len(did_fvid_count)  # 0.07208726739389495

    valid_candidate_vid = valid_click['vid'].unique()  # vid 14776

    # test part_test
    test_click = pd.read_csv(
        data_path + "/part_test/test_candidate_did_fvid.csv")  # 91741

    did_fvid_count = test_click.groupby(
        'did')['fvid'].count().reset_index()  # did 84076
    did_fvid_count['fvid'].describe()
    len(did_fvid_count[did_fvid_count['fvid'] > 1]) / \
        len(did_fvid_count)  # 0.07072172796041677

    test_candidate_vid = pd.read_csv(
        data_path + "/part_test/test_candidate_vid.csv")  # vid 17297

    #
    test_click['did'].nunique() * test_candidate_vid['vid'].nunique()

    # 候选vid在过去n天占比
    test_click = pd.read_csv(
        data_path + "/part_test/test_candidate_did_fvid.csv")  # 91741
    test_candidate_vid = pd.read_csv(
        data_path + "/part_test/test_candidate_vid.csv")  # vid 17297
    test_behavior = pd.read_csv(
        data_path + "/part_test/user_main_behavior.csv")
    test_behavior['date'] = test_behavior['timestamp'].apply(timestamp_to_date)

    tdate = test_behavior['date'].unique()
    tdate.sort()
    # temp = test_behavior[test_behavior['did'].isin(test_click['did'])]
    for i in (5, 10, 15, 30):
        temp = test_behavior[test_behavior['date'] >= tdate[-i]]
        vid = temp['vid'].drop_duplicates()
        print(len(vid[vid.isin(test_candidate_vid['vid'])]) /
              len(test_candidate_vid))

    # 0.8351737295484766
    # 0.9042608544834364
    # 0.928831589292941
    # 0.9532866971151067

    # 点击时间与创建时间差距
    test_click = pd.read_csv(
        data_path + "/part_test/test_candidate_did_fvid.csv")  # 91741
    test_candidate_vid = pd.read_csv(
        data_path + "/part_test/test_candidate_vid.csv")  # vid 17297
    vid = pd.concat([test_click['fvid'], test_candidate_vid['vid']])
    vid = pd.DataFrame(vid, columns=['vid'])
    vid['date'] = '2021-03-27'

    vid_info = pd.read_csv(data_path + "vid_info.csv")
    vid_info = vid_info[~vid_info['online_time'].isnull()]
    vid_info['online_time'] = vid_info['online_time'].apply(timestamp_to_date)

    vid = pd.merge(vid, vid_info, on='vid')
    vid['date'] = pd.to_datetime(vid['date'])
    vid['online_time'] = pd.to_datetime(vid['online_time'])
    vid['diff_day'] = vid['date'] - vid['online_time']
    vid['diff_day'] = vid['diff_day'].dt.days
    temp = vid.groupby('diff_day')['vid'].count().reset_index()
    temp['n'] = temp['vid'].expanding().sum()
    temp['ratio'] = temp['n']/temp['vid'].sum() * 100
    vid['diff_day'].plot(kind='kde')

    # test vid 点击人数
    test_click = pd.read_csv(
        data_path + "/part_test/test_candidate_did_fvid.csv")  # 91741
    test_candidate_vid = pd.read_csv(
        data_path + "/part_test/test_candidate_vid.csv")  # vid 17297
    test_behavior = pd.read_csv(
        data_path + "/part_test/user_main_behavior.csv")
    test_behavior = test_behavior[test_behavior['did'].isin(test_click['did'])]
    test_behavior = test_behavior[test_behavior['vid'].isin(
        test_candidate_vid['vid'])]
    temp = test_behavior.groupby('vid')['did'].nunique()
    temp.describe()
    # count    15412.00000
    # mean       162.50305
    # std        541.85857
    # min          1.00000
    # 25%          4.00000
    # 50%         15.00000
    # 75%         83.00000
    # max      14910.00000

    # 用户是否会重复观看视频
    df_click_data = pd.read_feather(base_path + 'dbfeed_click_info.feather')
    df_click_data['date'] = df_click_data['time'].apply(timestamp_to_date)

    t1 = df_click_data[['date', 'did', 'fvid', 'vts_ratio']].rename(columns={
                                                                    'fvid': 'vid'})
    t2 = df_click_data[['date', 'did', 'vid', 'vts_ratio']]
    t = pd.concat([t1, t2])

    tc = t[['did', 'vid']].value_counts().rename('times').reset_index()
    len(tc[tc['times'] == 1]) / len(tc)  # 0.8615 只看一次视频
    tc = tc[~tc['did'].isin(['46c0551471b8693aec72dd2a5df6f171'])]

    td = df_click_data[['did', 'fvid', 'vid', 'date', 'pos', 'pageNum', 'vts_ratio']
                       ][(df_click_data['did'] == 'c802affc6fe7d8011a8f11047961d778')
                         & ((df_click_data['fvid'] == 11779022) |
                         (df_click_data['vid'] == 11779022))]

    td = df_click_data[['did', 'fvid', 'vid', 'date', 'vts_ratio']
                       ][(df_click_data['did'] == 'c802affc6fe7d8011a8f11047961d778')]

    ts = df_show_data[df_show_data['did'] ==
                      '542f278aaecb202e14ae94626856eed7']
    tds = pd.merge(ts, td, on=['did', 'fvid', 'vid'], how='left')

    df_click_data.columns
    temp = df_click_data.head(100)

    # 点播信息流推荐模块点击日志
    df_click_data = pd.read_feather(base_path + 'dbfeed_click_info.feather')

    df_main_behavior = pd.read_pickle(base_path + 'user_main_behavior.pickle')

    # 点播信息流推荐模块曝光日志
    df_show_data = pd.read_feather(base_path + 'dbfeed_show_info.feather')

    # 保存视频信息表
    df_vid_info = pd.read_csv(base_path + "vid_info.csv")
    #df_vid_info['online_time'] = df_vid_info['online_time'].apply(lambda x : np.NaN if x <= 0 else x)

    # 保存视频明星表
    df_vid_tag_conf = pd.read_csv(base_path + "vid_stars_info.csv")

    # 保存视频标签信息表
    df_vid_dim_tags_conf = pd.read_csv(base_path + "vid_dim_tags_info.csv")

    # 保存标签信息表
    df_dim_tag_conf = pd.read_csv(base_path + "dim_tags_info.csv")

    df_click_data = df_click_data.merge(
        df_vid_info[['vid', 'cid']], on='vid', how='left')
    df_show_data = df_show_data.merge(
        df_vid_info[['vid', 'cid']], on='vid', how='left')

    df_show_data['date'] = df_show_data['time'].apply(
        timestamp_to_date).apply(lambda x: x.split(' ')[0])
    df_click_data['date'] = df_click_data['time'].apply(
        timestamp_to_date).apply(lambda x: x.split(' ')[0])
