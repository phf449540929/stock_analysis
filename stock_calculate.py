# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     stock_analyze.py
   Description :
   Author :      haifeng
   date：       2019/5/29 0029 上午 0:09
-------------------------------------------------
   Change Activity:
             2019/5/29 0029 上午 0:09
-------------------------------------------------
"""

import tushare
from pandas import DataFrame

"""
    tushare 是财经数据接口包，主要实现对股票等金融数据从数据采集、清洗加工到数据存储的过程。
    pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。
    pandas.DataFrame 是Python中Pandas库中的一种数据结构,它类似excel,是一种二维表。
"""

if __name__ == '__main__':
    # 先建立一个字典
    # 用来存储股票对应的价格
    stock_list = ['601398', '601939', '601857', '600028']
    """
        工商银行( 601398 )
        建设银行( 601939 )
        中国石油( 601857 )
        中国石化( 600028 )
    """
    all_price = {}
    # 遍历list里面的股票
    # 可以写入多个股票
    for ticker in stock_list:
        # 获取各股票某时段的价格
        all_price[ticker] = tushare.get_k_data(ticker, '2019-03-01', '2019-03-01')
        """
            get_k_data() 支持分时k线数据
            可替代以前的get_hist_data
            根据股票号码和日期获取股票数据
        """
    # 用for循环遍历股票价格并转换为dataframe的形式
    price = DataFrame({tic: data['close']
                       for tic, data in all_price.items()})
    """
        如果要同时得到关键词和值
        那么可以用items()来提取
    """
    # 计算股票价格每日变化
    returns = price.pct_change()
    """
        pct_change()计算百分数变化
        pct_change()默认遇到缺失值nan按照'pad'方法填充
    """
    # 计算相关性
    corr = returns.corr()
    """
        data.corr() 相关系数矩阵
        即给出了任意两个变量之间的相关系数
    """
    # 计算协方差
    cov = returns.cov()
    print("各股相关性")
    print(corr)
    print("各股协方差")
    print(cov)
