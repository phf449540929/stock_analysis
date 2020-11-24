# 研究方法论大作业

姓名：彭海峰

学号：20083900210005

班级：2020级网络空间安全硕士研究生

简介：

    对2019年11月1日至2020年11月1日的股票收盘价进行分析，并通过计算得到相关数据，以此绘制可视化图表。
    图表中的原始数据来源来自tushare库，tushare是一个财经数据接口包，主要实现对股票等金融数据从数据采集、清洗加工到数据存储的过程。
    得到了原始数据之后，在使用python进行画图之前，需要对得到的原始数据进行计算，这里使用了math、numpy、pandas、stockstats四个库进行计算。
    math 库提供了许多对浮点数的数学运算函数；numpy代表 “Numeric Python”，它是一个由多维数组对象和用于处理数组的例程集合组成的库；
    pandas 是基于NumPy 的个库，该工具是为了解决数据分析任务而创建的；
    stockstats是一个专用于对股票进行分析的库，其包含计算股票中的16个常用指标方法。
    最后画图使用了matplotlib库，这个库是Python中最常用的可视化工具之一，功能非常强大，
    而其中的matplotlib.pyplot 是一个命令型函数集合，它可以让我们像使用MATLAB一样使用matplotlib。


## 目录

<ul>
    <li>研究方法论大作业	1</li>
    <li>（一）获取股票原始数据	3</li>
    <li>（二）股票收盘价折线图	3</li>
    <li>（三）股票收盘价差折线图	5</li>
    <li>（四）股票收益价格折线图	7</li>
    <li>（五）股票收益率移动历史标准差折线图	8</li>
    <li>（六）股票外汇量差分析指标折线图	10</li>
    <li>（七）股票向前与向后n天折线图	13</li>
    <li>（八）股票n天跌涨百分比折线图	15</li>
</ul>

## （一）获取股票原始数据

    首先我们要从tushare库中获取股票的原始数据，
    这里我选取了从去年，也就是2019年11月1日，至今年，也就是2020年11月1日的股票数据。

获取股票原始数据的核心代码如下：

```
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
```

## （二）股票收盘价折线图

    在获取到近一年的股票数据之后，我们便需要对股票的数据依次进行计算与画图。
    首先分别以5天、10天、20天、60天为周期对股票收盘价进行计算与绘图。

绘制股票收盘价折线图的核心代码如下：

```
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
```

股票收盘价折线图如图1所示：

![blockchain](./result/(1)line_chart_of_stock_closing.png "图1 股票收盘价折线图")
图1 股票收盘价折线图

## （三）股票收盘价差折线图

    之后便对股票收盘价差折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价，使得图像更加直观。
    股票收盘价差分为周-半月收盘价差与周-月收盘价差。
    
绘制股票收盘价差折线图的核心代码如下：

```
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
```

股票收盘价差折线图如图2所示：

![blockchain](./result/(2)line_chart_of_stock_closing_price_difference.png "图2 股票收盘价差折线图")
图2 股票收盘价差折线图

## （四）股票收益价格折线图

    之后便对股票收益价格折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价，使得图像更加直观。
    
绘制股票收益价格折线图的核心代码如下：

```
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
```

股票收益价格折线图如图3所示：

![blockchain](./result/(3)line_chart_of_stock_return_price.png "图3 股票收益价格折线图")
图3 股票收益价格折线图

## （五）股票收益率移动历史标准差折线图

    之后便对股票收益率移动历史标准差折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价、股票收益价格，使得图像更加直观。
    
绘制股票收益率移动历史标准差折线图的核心代码如下：

```
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
```

股票收益率移动历史标准差折线图如图4所示：

![blockchain](./result/(4)line_chart_of_stock_return_moving_historical_standard_deviation.png "图4 股票收益率移动历史标准差折线图")
图4 股票收益率移动历史标准差折线图

## （六）股票外汇量差分析指标折线图

    之后便对股票外汇量差分析指标折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价，使得图像更加直观。
    除此之外对股票外汇量差分析指标与其前一天的体积增量进行计算与绘制，
    通过绘制两个图像，可以使股票外汇量差分析指标更加清晰。
    
绘制股票外汇量差分析指标折线图的核心代码如下：

```
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
```

    在这里用两个图来绘制股票外汇量差分析指标折线图，
    第一个为股票外汇量差分析指标与股票外汇量与其前一天的体积增量，
    第二个为股票外汇量差分析指标与股票收盘价的对比。
    通过这两个图便可以更清晰的观察股票外汇量差分析指标。
    
股票外汇量差分析指标折线图如图5、图6所示：

![blockchain](./result/(5)line_chart_of_stock_foreign_exchange_volume_difference_analysis_index_in_lastday.png "图5 股票外汇量差分析指标折线图")
图5 股票外汇量差分析指标折线图

![blockchain](./result/(6)line_chart_of_stock_foreign_exchange_volume_difference_analysis_index_in_closing.png "图6 股票外汇量差分析指标折线图")
图6 股票外汇量差分析指标折线图

## （七）股票向前与向后n天折线图

    之后便对股票向前与向后n天折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价，使得图像更加直观。
    为使图像更加简洁，只选取股票的向前2天与向后2天的范围。
    
绘制股票向前与向后n天折线图的核心代码如下：

```
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
```

股票向前与向后n天折线图如图7所示：

![blockchain](./result/(7)line_chart_of_stock_forward_and_backward_n_day.png "图7 股票向前与向后n天折线图")
图7 股票向前与向后n天折线图

## （八）股票n天跌涨百分比折线图

    之后便对股票n天跌涨百分比折线图进行计算与绘制，
    在此为了便于进行对比，同样绘制股票收盘价，使得图像更加直观。
    为使图像更加简洁，只选取股票的2天内跌涨为范围。
    
绘制股票n天跌涨百分比折线图的核心代码如下：

```
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
```

股票n天跌涨百分比折线图如图8所示：

![blockchain](./result/(8)line_chart_of_stock_percentage_up_down.png "图8 股票n天跌涨百分比折线图")
图8 股票n天跌涨百分比折线图
