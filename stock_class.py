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

import numpy
import pandas
import tushare
from matplotlib import pyplot
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.cluster import affinity_propagation
# from sklearn.covariance import GraphLassoCV
from sklearn.covariance import GraphicalLassoCV

"""
    numpy 代表 “Numeric Python”，它是一个由多维数组对象和用于处理数组的例程集合组成的库。
    pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。
    tushare 是财经数据接口包，主要实现对股票等金融数据从数据采集、清洗加工到数据存储的过程。
    matplotlib 是Python中最常用的可视化工具之一，功能非常强大。
    matplotlib.pyplot 是一个命令型函数集合，它可以让我们像使用MATLAB一样使用matplotlib。
    matplotlib.collections 是Python内建的一个集合模块,提供了许多有用的集合类。
    matplotlib.collections.LineCollection实现在图形中绘制多条线，作为面向对象绘图的一部分。
    sklearn 是机器学习中常用的第三方模块,对常用的机器学习方法进行了封装。
    sklearn.manifold 为流形学习，是一种非线性降维的手段。
    sklearn.cluster 模块提供各聚类算法函数可以使用不同的数据形式作为输入。
    sklearn.cluster.affinity_propagation 为AP聚类算法，是基于数据点间的"信息传递"的一种聚类算法。
    sklearn.covariance 模块提供了几种协方差估计方法。
    sklearn.covariance.GraphLassoCV 得到股票原始数据之间的相关性图。
"""

"""
    对一系列股票数据聚类分析-近邻传播算法
"""


# 对这些结果进行可视化
def visual_stock_relationship(dataset, edge_model, labels, stock_names):
    """
    可视化结果
    :param dataset: 数据集
    :param edge_model: 模型
    :param labels: 标签
    :param stock_names:股票名称
    :return: none: 无
    """

    # LocallyLinearEmbedding LLE降维
    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, n_neighbors=6, eigen_solver='dense')
    """
        n_components 降维到的维数
        n_neighbors 搜索样本的近邻的个数,越大，降纬后的局部数据越好
        eigen_solver 特征分解的方法。有‘arpack’和‘dense’两者算法选择
    """
    # 处理数据
    embedding = node_position_model.fit_transform(dataset.T).T
    """
        fit_transform()是对数据进行某种统一处理
        比如标准化~N(0,1)
        将数据缩放(映射)到某个固定区间
        归一化
        正则化等
    """

    # 画出图表
    figure = pyplot.figure(1, facecolor='w', figsize=(10, 8))
    # 清除所有轴
    pyplot.clf()
    """
        清除所有轴但是窗口打开
        这样它可以被重复使用。
    """
    # 生成子图
    axe = pyplot.axes([0., 0., 1., 1.])
    # 设置轴属性
    pyplot.axis('off')

    # 显示偏相关图
    partial_correlations = edge_model.precision_.copy()
    d = 1 / numpy.sqrt(numpy.diag(partial_correlations))
    """
        np.sqrt() 开根号
        numpy.diag()返回一个矩阵的对角线元素
        或者创建一个对角阵（ diagonal array）
    """
    partial_correlations *= d
    partial_correlations *= d[:, numpy.newaxis]
    """
        numpy.newaxis从字面上来理解就是用来创建新轴的
        或者说是用来对array进行维度扩展的。
    """
    non_zero = (numpy.abs(numpy.triu(partial_correlations, k=1)) > 0.02)
    """
        numpy.abs() 计算数组各元素的绝对值
        numpy.triu() 与tril类似,返回的是矩阵的上三角矩阵
    """

    # 使用嵌入的坐标绘制节点
    pyplot.scatter(
        embedding[0], embedding[1], s=100 * d ** 2, c=labels, cmap=pyplot.cm.nipy_spectral)
    """
        pyplot.scatter() 画散点图
    """

    # 绘制边缘
    start_idx, end_idx = numpy.where(non_zero)
    """
        numpy.where() 输出满足条件 (即非0) 元素的坐标 
        等价于numpy.nonzero
    """
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [
        [embedding[:, start], embedding[:, stop]]
        for start, stop in zip(start_idx, end_idx)
    ]
    values = numpy.abs(partial_correlations[non_zero])
    # 绘制LineCollection曲线
    lc = LineCollection(
        segments, zorder=0, cmap=pyplot.cm.hot_r, norm=pyplot.Normalize(0, .7 * values.max()))
    """
        LineCollection实现在图形中绘制多条线
        作为面向对象绘图的一部分。
    """
    # 将LineCollection曲线添加到子图中
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    axe.add_collection(lc)

    # 向每个节点添加标签
    # 放置标签以避免与其他标签重叠
    n_labels = max(labels)
    for index, (name, label, (x, y)) in enumerate(
            zip(stock_names, labels, embedding.T)
    ):

        # 计算坐标
        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[numpy.argmin(numpy.abs(dy))]
        this_dy = dy[numpy.argmin(numpy.abs(dx))]

        # 根据其位置调整方向
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .001
        else:
            horizontalalignment = 'right'
            x = x - .001
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .001
        else:
            verticalalignment = 'top'
            y = y - .001

        pyplot.text(x, y, name, size=10, fontproperties='SimHei',
                    horizontalalignment=horizontalalignment,
                    verticalalignment=verticalalignment,
                    bbox=dict(facecolor='w',
                              edgecolor=pyplot.cm.nipy_spectral(label / float(n_labels)),
                              alpha=.6))
        """
            pyplot.text()添加文本信息
        """

    pyplot.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
                embedding[0].max() + .10 * embedding[0].ptp(), )
    pyplot.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
                embedding[1].max() + .03 * embedding[1].ptp())
    """
        pyplot.xlim() 设定横坐标的上下限
        pyplot.ylim() 设定纵坐标的上下限
    """
    pyplot.show()


def get_k_dataframe(code, start, end):
    """
    获取k天数据
    从开始日期到结束日期
    :param code: 股票的编号
    :param start: 开始日期
    :param end: 结束日期
    :return: DataFrame: 获得的数据[date, open, close, high, low]
    """

    # 获得股票的数据
    df = tushare.get_k_data(code, start=start, end=end)
    """
        get_k_data() 支持分时k线数据
        可替代以前的get_hist_data
        根据股票号码和日期获取股票数据
     """
    # 清洗数据
    df.drop(['volume'], axis=1, inplace=True)
    """
        df.drop() 删除一行或一列数据
    """
    return df


def get_batch_k_df(codes_list, start, end):
    """
    获取批量库存K数据
    :param codes_list: 股票编号的列表
    :param start: 开始日期
    :param end: 结束日期
    :return: DataFrame: 获得的数据[date, open, close, high, low]
    """

    df = pandas.DataFrame()
    print('fetching data. pls wait...')
    # 通过遍历股票编号的列表循环获取股票数据
    for code in codes_list:
        # print('fetching K data of {}...'.format(code))
        df = df.append(get_k_dataframe(code, start, end))
        """
            DataFrame.append() 将其他行添加到此DataFrame的末尾
            返回一个新对象
            get_k_data() 支持分时k线数据
            可替代以前的get_hist_data
            根据股票号码和日期获取股票数据
        """
    return df


def preprocess_data(stock_df, min_K_num=1000):
    """
    预处理库存数据
    :param stock_df: 股票的DataFrame
    :param min_K_num: 最小库存K号
    :return: stock_dataset: 数据集
            result_df.columns.tolist(): 预处理的结果
    """

    df = stock_df.copy()
    # 此处用收盘价与开盘价的差值做分析
    df['diff'] = df.close - df.open
    # 清洗数据
    df.drop(['open', 'close', 'high', 'low'], axis=1, inplace=True)

    result_df = None
    # 下面一部分是将不同的股票diff数据整合为不同的列
    # 列名为股票代码
    for name, group in df[['date', 'diff']].groupby(df.code):
        if len(group.index) < min_K_num: continue
        if result_df is None:
            result_df = group.rename(columns={'diff': name})
        else:
            result_df = pandas.merge(
                result_df, group.rename(columns={'diff': name}),
                on='date', how='inner')
            """
                一定要inner
                不然会有很多日期由于股票停牌没数据
            """

    result_df.drop(['date'], axis=1, inplace=True)
    # 然后将股票数据DataFrame转变为np.ndarray
    stock_dataset = numpy.array(result_df).astype(numpy.float64)
    """
        numpy.array() numpy中array类矩阵相关方法
    """
    # 数据归一化
    stock_dataset /= numpy.std(stock_dataset, axis=0)
    """
        此处使用相关性而不是协方差的原因是在结构恢复时更高效
    """
    return stock_dataset, result_df.columns.tolist()


"""
    上面准备了各种函数
    下面开始准备数据集
    此处分析上证50指数的成分股
    分析这些股票有哪些特性
"""

sz50_df = tushare.get_sz50s()
print(sz50_df)
stock_list = sz50_df.code.tolist()
# print(stock_list)
# 没有问题
# 查看最近五年的数据
batch_K_data = get_batch_k_df(stock_list, start='2014-05-01', end='2019-05-01')
print(batch_K_data.info())

# (603, 41) 由此可以看出得到了603个交易日的数据
# 其中有41只股票被选出。
stock_dataset, selected_stocks = preprocess_data(batch_K_data, min_K_num=1100)
print(stock_dataset.shape)
"""
    其他的9只股票因为不满足最小交易日的要求而被删除
    这603个交易日是所有41只股票都在交易
    都没有停牌的数据。
"""
# 这是实际使用的股票列表
print(selected_stocks)

# 对这41只股票进行聚类
edge_model = GraphicalLassoCV()
edge_model.fit(stock_dataset)
_, labels = affinity_propagation(edge_model.covariance_)
n_labels = max(labels)
"""
    labels里面是每只股票对应的类别标号  
"""
print('Stock Clusters: {}'.format(n_labels + 1))
"""
    10
    即得到10个类别
"""
sz50_df2 = sz50_df.set_index('code')
# print(sz50_df2)
for i in range(n_labels + 1):
    # print('Cluster: {}----> stocks: {}'.format(i,','.join(np.array(selected_stocks)[labels==i])))
    """
        这个只有股票代码而不是股票名称
        下面打印出股票名称
        便于观察
    """
    stocks = numpy.array(selected_stocks)[labels == i].tolist()
    """
        numpy.array() numpy中array类矩阵相关方法
    """
    names = sz50_df2.loc[stocks, :].name.tolist()
    print('Cluster: {}----> stocks: {}'.format(i, ','.join(names)))

stock_names = sz50_df2.loc[selected_stocks, :].name.tolist()
visual_stock_relationship(stock_dataset, edge_model, labels, stock_names)
