# MGTV_AI_Challenge_Video_Recommend_Rank_12th

&nbsp;

# 赛题介绍
2021年芒果TV第二届“马栏山杯”国际音视频算法大赛视频推荐

设法提高视频推荐点击率以及人均有效观看时长，是芒果TV平台的核心技术挑战之一，本赛题以芒果TV点播信息流模块的实际推荐业务场景为原型，打造了一个经典的推荐问题，并且主办方提供了丰富的特征维度以及海量的数据信息内容，希望选手设计出一套精准有效的推荐模型，以探索进一步提升视频推荐效果的技术路径。

给定点播信息流模块的候选(用户-触发视频)集合S以及候选视频集合V, 从V中生成每个(用户-触发视频) s∈S最有可能点击的N个视频，其中N=30，以及预测s∈S观看列表中每个视频的播放时长比例。

比赛网址：https://challenge.ai.mgtv.com/contest/detail/9

&nbsp;

视频推荐数据说明:

    1.part_1 - part_7为1-7天的历史行为数据，其中每个part包含三个文件dbfeed_click_info.csv, dbfeed_show_info.csv, user_main_behavior.csv

        (1):dbfeed_click_info.csv为点播信息流推荐模块点击日志
        (2):dbfeed_show_info.csv为点播信息流推荐模块曝光日志
        (3):user_main_behavior.csv为当天点播信息流用户过去90天的主站观看行为序列

    2.part_test为第八天的数据，包含test_candidate_did_fvid.csv, test_candidate_vid.csv, user_main_behavior.csv
        (1):test_candidate_did_fvid.csv为候选用户-触发视频(测试数据),表示当天did在fvid下有过点击的所有候选did-fvid集合
        (2):test_candidate_vid.csv为候选vid(测试数据),表示当天所有有过点击记录的vid集合
        (3):user_main_behavior.csv为当天点播信息流用户过去90天的主站观看行为序列  

    3.vid_info.csv为视频信息表
      vid_stars_info.csv为视频明星信息表
      vid_dim_tags_info.csv为视频标签信息表
      dim_tags_info.csv为标签信息表
          
A榜为第八天的数据，B榜为第九天的数据，A榜结束时，会发布第八天的数据

&nbsp;    
# 模型介绍
### 召回规则：
 - 过N天fvid下，vid的曝光率的TOPN进行召回
 - 过N天fvid下，vid的观看比例的TOPN进行召回
 - 过N天cid下，vid的观看比例的TOPN进行召回
 - 过N天cid下，vid的最新上线日期的TOPN进行召回
 - 过N天的热门(vid的点击次数，观看比例的TOPN进行召回)

### 排序模型：
 - fvid与vid是否属于同一cid
 - vid最后一次点击时间差、上线天数
 - did分组统计fvid、vid、cid的点击nunique、count
 - did分组统计fvid、vid、cid的曝光nunique、count
 - did与fvid、vid、cid的组合点击count、观看总时长、平均观看时长
 - fvid与vid、cid的组合点击count、观看总时长、平均观看时长
 - 当天did观看cid的次数
 - 前七天做特征工程，单日训练LightGBM模型预测下一日

&nbsp;

# 代码注释

```python

# 召回规则
recall_ruler.py

# 特征工程
feature_engineering.py

# 模型训练
main.py
```

