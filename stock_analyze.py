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

    # CR指标
    stockStat[
        ['close', 'cr', 'cr-ma1', 'cr-ma2', 'cr-ma3']
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
        "CR指标",
        "周CR指标",
        "半月CR指标",
        "月CR指标"
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
        CR指标又称价格动量指标
        CR跌穿a、b、c、d四条线再由低点向上爬升160时
        为短线获利的一个良机
        应适当卖出股票
        CR跌至40以下时是建仓良机
        而CR高于300~400时应注意适当减仓
        CR指示器平均移动了5、10、20天
    """

    # KDJ指标
    # 分别是k d j 三个数据统计项。
    stockStat[
        ['close', 'kdjk', 'kdjd', 'kdjj']
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
        "KDJK指标",
        "KDJD指标",
        "KDJJ指标"
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
        随机指标(KDJ)
        一般是根据统计学的原理
        通过一个特定的周期（常为9日、9周等）内出现过的最高价、最低价
        和最后一个计算周期的收盘价及这三者之间的比例关系
        来计算最后一个计算周期的未成熟随机值RSV
        然后根据平滑移动平均线的方法来计算K值、D值与J值
        绘成曲线图来研判股票走势
    """
    # 3天Kdjk交叉3天Kdjd
    # stockStat['kdjk_3_xu_kdjd_3'].plot(figsize=(20,10), grid=True)
    # pyplot.show()
    # print(stockStat['kdjk_3_xu_kdjd_3'].tail())

    # SMA指标
    stockStat[
        ['close', 'close_5_sma', 'close_10_sma']
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
        "周SMA指标",
        "半月SMA指标"
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
        简单移动平均线（Simple Moving Average）
        简称SMA指标
        可以动态输入参数
        获得几天的移动平均
    """

    # MACD指标
    stockStat[
        ['close', 'macd', 'macds', 'macdh']
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
        "MACD指标",
        "MACDS指标",
        "MACDH指标"
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
        平滑异同移动平均线(Moving Average Convergence Divergence)
        简称MACD指标
        也称移动平均聚散指标
        MACD技术分析运用DIF线与MACD线之相交型态及直线棒高低点与背离现象作为买卖讯号
        尤其当市场股价走势呈一较为明确波段趋势时MACD则可发挥其应有的功能
        但当市场呈牛皮盘整格局股价不上不下时MACD买卖讯号较不明显 
    """

    # BOLL指标
    stockStat[
        ['close', 'boll', 'boll_ub', 'boll_lb']
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
        "BOLL指标",
        "BOLLUB指标",
        "BOLLLB指标"
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
        布林线指标(Bollinger Bands)
        当布林线开口向上后
        只要股价K线始终运行在布林线的中轨上方的时候
        说明股价一直处在一个中长期上升轨道之中
        这是BOLL指标发出的持股待涨信号
        如果TRIX指标也是发出持股信号时
        这种信号更加准确
        此时投资者应坚决持股待涨
    """

    # RSI指标
    stockStat[
        ['close', 'rsi_6', 'rsi_12']
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
        "周RSI指标",
        "半月RSI指标"
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
        相对强弱指标（Relative Strength Index）
        简称RSI
        也称相对强弱指数、相对力度指数
        强弱指标保持高于50表示为强势市场
        反之低于50表示为弱势市场
    """

    # WR指标
    stockStat[
        ['close', 'wr_6', 'wr_10']
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
        "周WR指标",
        "半月WR指标"
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
        威廉指数（Williams % Rate）
        该指数是利用摆动点来度量市场的超买超卖现象
    """

    # CCI指标
    stockStat[
        ['close', 'cci', 'cci_20']
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
        "CCI指标",
        "半月CCI指标"
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
        顺势指标又叫CCI指标
        其英文全称为“Commodity Channel Index”
        是由美国股市分析家唐纳德·蓝伯特（Donald Lambert）所创造的
        是一种重点研判股价偏离度的股市分析工具
        CCI默认为14天的指标
        在这里使用20天的CCI指标
    """

    # ATR指标
    stockStat[
        ['close', 'tr', 'atr']
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
        "TR指标",
        "ATR指标"
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
        均幅指标（ATR）
        是取一定时间周期内的股价波动幅度的移动平均值
        主要用于研判买卖时机
        均幅指标无论是从下向上穿越移动平均线
        还是从上向下穿越移动平均线时
        都是一种研判信号
    """

    # DMA指标
    stockStat[
        ['close', 'dma']
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
        "DMA指标"
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
        DMA指标（Different of Moving Average）
        又叫平行线差指标
        是目前股市分析技术指标中的一种中短期指标
        它常用于大盘指数和个股的研判
    """

    # DMI，+DI，-DI，DX，ADX，ADXR指标
    stockStat[
        ['close', 'pdi', 'mdi', 'dx', 'adx', 'adxr']
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
        "PANDASI指标",
        "MDI指标",
        "DX指标",
        "ADX指标",
        "ADXR指标"
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
        动向指数Directional Movement Index,DMI）
        平均趋向指标（Average Directional Indicator，简称ADX）
        平均方向指数评估（ADXR）实际是今日ADX与前面某一日的ADX的平均值
        ADXR在高位与ADX同步下滑可以增加对ADX已经调头的尽早确认
        ADXR是ADX的附属产品
        只能发出一种辅助和肯定的讯号
        并非入市的指标
        而只需同时配合动向指标(DMI)的趋势才可作出买卖策略。
    """

    # TRIX指标
    stockStat[
        ['close', 'trix', 'trix_9_sma']
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
        "TRIX指标",
        "TRIXSMA指标"
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
        TRIX指标（Triple Exponentially Smoothed Average）
        又叫三重指数平滑移动平均指标
    """

    # VR指标
    stockStat[
        ['close', 'vr', 'vr_6_sma']
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
        "VR指标",
        "VRSMA指标"
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
        成交量比率（Volumn Ratio，VR）
        简称VR
        是一项通过分析股价上升日成交额（或成交量）与股价下降日成交额比值
        从而掌握市场买卖气势的中期技术指标
    """
