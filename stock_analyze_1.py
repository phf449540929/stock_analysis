# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: haifeng
@Time: 2020/11/23 22:27
@File: stock_analyze.py
@Since: python 3.9
@Version: V 1.0
@Description: 研究方法论大作业
@Change:
    haifeng 2020/11/23 22:27 创建文件
"""

import math
import numpy
import pandas
import tushare
import stockstats
from matplotlib import pyplot

"""
    math 模块提供了许多对浮点数的数学运算函数。
    numpy 代表 “Numeric Python”，它是一个由多维数组对象和用于处理数组的例程集合组成的库。
    pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。
    tushare 是财经数据接口包，主要实现对股票等金融数据从数据采集、清洗加工到数据存储的过程。
    stockstats 包含计算股票中的16个常用指标方法。
    matplotlib 是Python中最常用的可视化工具之一，功能非常强大。
    matplotlib.pyplot 是一个命令型函数集合，它可以让我们像使用MATLAB一样使用matplotlib。
"""

# 使得每次运行得到的随机数都一样
numpy.random.seed(37)
"""
    如果在seed()中传入的数字相同
    那么接下来使用random()或者rand()方法所生成的随机数序列都是相同的
"""
# 一系列包的输出显示设置
pandas.set_option('display.max_columns', None)
# 显示所有行
pandas.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pandas.set_option('max_colwidth', 100)
# 用来正常显示中文标签
pyplot.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
pyplot.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 设置分析的股票的时间段
    begin_time = '2019-11-01'
    end_time = '2020-11-01'
    # 设置编号
    code = "000001"
    # 获取到目前为止3年的历史数据
    stock = tushare.get_hist_data(code, start=begin_time, end=end_time)
    """
        get_hist_data这个接口只能获取近3年的日线数据
        适合搭配均线数据进行选股和分析
    """
    # 将数据按照日期排序
    stock = stock.sort_index(0)
    """
        sort_index()默认根据行标签对所有行排序
        或根据列标签对所有列排序
        或根据指定某列或某几列对行排序。
    """
    # 生成pickle文件
    stock.to_pickle("stock_data_000001.pickle")
    """
        to_pickle()可以生成pickle文件对数据进行永久储存
    """
    # 基础股票数据
    print("finish save ...")
    print(stock)

    # 读取股票数据分析。不用每次网络请求数据
    stock = pandas.read_pickle("stock_data_000001.pickle")
    """
        read_pickle()读取pickle数据
    """
    print(type(stock))

    # 周线
    stock["5d"] = stock["close"].rolling(window=5).mean()
    # 半月线
    stock["10d"] = stock["close"].rolling(window=10).mean()
    # 月线
    stock["20d"] = stock["close"].rolling(window=20).mean()
    # 季度线
    stock["60d"] = stock["close"].rolling(window=60).mean()
    """
        5周期、10周期、20周期和60周期
        周线、半月线、月线和季度线
    """
    # print(stock.head(1))
    # 展示股票收盘价格信息
    stock[
        ["close", "5d", "10d", "20d", "60d", ]
    ].plot(figsize=(20, 10), fontsize=16, grid=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
    """
    # 设置标题
    pyplot.title("股票收盘价折线图", fontsize=24)
    # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 设置Y轴标签
    pyplot.ylabel("收盘价", fontsize=24)
    # 设置图例字体大小
    pyplot.legend(fontsize=16)
    # 展示图表
    pyplot.show()
    """
        股票收盘价的曲线图
    """

    # 周-半月线差
    stock["5-10d"] = stock["5d"] - stock["10d"]
    # 周-月线差
    stock["5-20d"] = stock["5d"] - stock["20d"]
    # 展示周，半月收盘价差
    stock[
        ["close", "5-10d", "5-20d"]
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "收盘价",
        "周-半月收盘价差",
        "周-月收盘价差"
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
    # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        周、半月收盘价差
    """

    # 计算股票的收益价格
    stock["return"] = numpy.log(stock["close"] / stock["close"].shift(1))
    # stock["return_a"] = stock["close"] / stock["close"].shift(1)
    # print(stock[["return","return_a"]].head(15))
    # 展示股票的收益价格
    stock[
        ["close", "return"]
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "收盘价",
        "收益价格"
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
        # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        股票的收益价格
    """

    # 计算股票的收益率移动历史标准差
    mov_day = int(len(stock) / 20)
    stock["mov_vol"] = stock["return"].rolling(window=mov_day).std() * math.sqrt(mov_day)
    # print(stock["mov_vol"].head(mov_day+1))
    # 展示股票的收益率移动历史标准差
    stock[
        ["close", "mov_vol", "return"]
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "收盘价",
        "收益率移动历史标准差",
        "收益价格"
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
        # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        股票的收益率移动历史标准差
    """

    # 计算股票的16个指标
    # 将提取的股票数据转换成另一个股票的类
    stockStat = stockstats.StockDataFrame.retype(stock)
    """
        初始化StockDataFrame
        使用retype函数
        将Pandas.DataFrame转到StockDataFrame
    """

    # 外汇量差分析指标(VolumeDelta)
    # volume_delta是与前一天相比的体积增量
    # 单位为(Vol ∆)
    stockStat[
        ['volume', 'volume_delta']
    ].plot(figsize=(20, 10), fontsize=16, grid=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
    """
    # 设置标题
    pyplot.title("股票外汇量差分析指标折线图", fontsize=24)
    # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 设置Y轴标签
    pyplot.ylabel("外汇量差分析指标", fontsize=24)
    # 设置图例字体大小
    pyplot.legend(fontsize=16)
    # 展示图表
    pyplot.show()
    """
        交易量的delta转换
        交易量是正
        volume_delta把跌变成负值
    """
    stockStat[
        ['close', 'close_delta']
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "收盘价",
        "外汇量差分析指标",
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
        # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        股票的外汇量差分析指标
    """

    # 计算n天差
    stockStat[
        ['close_-2_d', 'close_-1_d', 'close', 'close_1_d', 'close_2_d']
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "向前2天差",
        "向前1天差",
        "收盘价",
        "向后1天差",
        "向后2天差",
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
        # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        可以计算向前n天和向后n天的差
        直接使用 key “n_d” 或 “-n_d”
        向前和向后对涨跌的趋势判断不太一样
        使用”_-n_d” 比较像原始数据的涨跌
    """

    # 计算n天涨跌百分百
    stockStat[
        ['close', 'close_-1_r', 'close_-2_r']
    ].plot(figsize=(20, 10), fontsize=16, grid=True, subplots=True)
    """
        figsize 图表大小
        fontsieze 坐标轴字体大小
        grad 是否出现网格
        subplots 是否分割图表
    """
    # 获取所有的子图
    axes = pyplot.gcf().axes
    # 子图标题的列表
    titles = [
        "收盘价",
        "1天涨跌百分比",
        "2天涨跌百分比"
    ]
    # 循环对每一个子图进行操作
    for i in range(0, len(axes)):
        # 设置子图标题
        axes[i].set_title("股票" + titles[i] + "折线图", fontsize=24)
        # 设置子图Y轴标签
        axes[i].set_ylabel(titles[i], fontsize=24)
        # 设置子图Y轴标签的偏移量
        axes[i].yaxis.set_label_coords(-0.04, 0.5)
        # 设置子图图例的字体大小
        axes[i].legend(fontsize=16)
        # 设置X轴标签
    pyplot.xlabel("日期", fontsize=24)
    # 展示图表
    pyplot.show()
    """
        n天涨跌百分百
    """

