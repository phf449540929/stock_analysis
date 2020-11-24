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
from pmdarima import auto_arima
from fastai.tabular import add_datepart
from matplotlib import pyplot
from matplotlib.pylab import rcParams
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

"""
    numpy 代表 “Numeric Python”，它是一个由多维数组对象和用于处理数组的例程集合组成的库。
    pandas 是基于NumPy 的一种工具，该工具是为了解决数据分析任务而创建的。
    pmdarima 通过包装操作创建一个更方便用户的评估器界面，供熟悉Scikit学习的程序员使用。
    pmdarima.auto_arima 构建高性能时间序列模型。
    fastai 库利用现代最佳实践简化了快速和准确的神经网络训练。
    fastai.tabular 模块来设置和训练模型。 
    fastai.tabular.add_datepart 函数来为我们创建这些特征。
    matplotlib 是Python中最常用的可视化工具之一，功能非常强大。
    matplotlib.pyplot 是一个命令型函数集合，它可以让我们像使用MATLAB一样使用matplotlib。
    matplotlib.pylab 包括了许多NumPy和pyplot模块中常用的函数，方便用户快速进行计算和绘图。
    matplotlib.pylab.rcParams 使用rc配置文件来自定义图形的各种默认属性。
    keras 代码包提供了一些更高级的模型，是一个高层神经网络API。
    keras.layers 该模块主要用于生成神经网络层，包含多种类型。
    keras.layers.LSTM 循环神经网络的lstm层。
    keras.layers.Dense 全连接层，对上一层的神经元进行全部连接，实现特征的非线性组合。
    sklearn 是机器学习中常用的第三方模块,对常用的机器学习方法进行了封装。
    sklearn.model_selection 预处理模块
    sklearn.model_selection.GridSearchCV 只要把参数输进去，就能给出最优化的结果和参数。
    sklearn.preprocessing 预处理模块
    sklearn.preprocessing.MinMaxScaler 原数据-最大/最大-最小，只要知道最大，最小数据。
    sklearn.linear_model 线性模型
    sklearn.linear_model 中文叫做线性回归,是一种基础、常用的回归方法。
"""

# 设置figure的尺寸
rcParams['figure.figsize'] = 20, 10
# 用于规范化数据
scaler = MinMaxScaler(feature_range=(0, 1))

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

# 读取文件
df = pandas.read_csv('NSE-TATAGLOBAL11.csv')
# 展示头部数据
print(df.head())

# 画出目标变量函数
# 就相当于待预测值
df['Date'] = pandas.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']
# 设置图表尺寸
pyplot.figure(figsize=(20, 10))
pyplot.plot(df['Close'], label='收盘价历史数据')
# 设置标题
pyplot.title("收盘价曲线", fontsize=24)
# 设置图例
pyplot.legend("收盘价曲线", fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()

# 移动平均法拆分数据集验证集
data = df.sort_index(ascending=True, axis=0)
new_data = pandas.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# 获取数据集
train = new_data[:987]
"""
    因为是时间序列
    不采用随机分割
    前4年训练
    去年验证
"""
# 预测误差
valid = new_data[987:]
"""
    为验证集创建预测值
    使用真实值检查RMSE误差
"""

preds = []
for i in range(0, 248):
    a = train['Close'][len(train) - 248 + i:].sum() + sum(preds)
    b = a / 248
    preds.append(b)

rms = numpy.sqrt(numpy.mean(numpy.power((numpy.array(valid['Close']) - preds), 2)))
print(rms)

# 建立评估方法
valid['Predictions'] = 0
valid['Predictions'] = preds
# 绘制曲线
pyplot.plot(train['Close'])
pyplot.plot(valid[['Close', 'Predictions']])
# 设置标题
pyplot.title("误差预测曲线", fontsize=24)
# 设置图例
pyplot.legend(["原始曲线", "真实曲线", "预测曲线"], fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()
"""
    评估方法建立完成
    可以看到我们如今的预测值算出的指标与真实值差距很大
    接下来使用不同种类机器学习
"""

# 将索引设置为日期值
df['Date'] = pandas.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# 升序对数据排序
# 创造新数据集
data = df.sort_index(ascending=True, axis=0)
# 创建单独的数据集
new_data = pandas.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
print(data)
# 分析特征
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    """
        我们将表里的特征分的更细
        创造出了新特征
        ‘Year’,
        ‘Month’,
        ‘Week’,
        ‘Day’,
        ‘Dayofweek’,
        ‘Dayofyear’,
        ‘Is_month_end’,
        ‘Is_month_start’,
        ‘Is_quarter_end’,
        ‘Is_quarter_start’, 
        ‘Is_year_end’,
        ‘Is_year_start’
    """

# 进行相关性预测
add_datepart(new_data, 'Date')
# 时间戳
new_data.drop('Elapsed', axis=1, inplace=True)
"""
    考虑到股票在每周前几天交易量最大
    将每一天与其他天进行相关性预测
"""
new_data['mon_fri'] = 0
for i in range(0, len(new_data)):
    if new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4:
        new_data['mon_fri'][i] = 1
    else:
        new_data['mon_fri'][i] = 0
    """
        星期一/星期五还是星期二/星期三/星期四
        这两种相关性给出1或者0
    """

# 创建数据集
train = new_data[:987]
valid = new_data[987:]
x_train = train.drop('Close', axis=1)
y_train = train['Close']
x_valid = valid.drop('Close', axis=1)
y_valid = valid['Close']
"""
    数据集创建完毕
"""

# 线性回归
model = LinearRegression()
model.fit(x_train, y_train)
preds = model.predict(x_valid)
rms = numpy.sqrt(numpy.mean(numpy.power((numpy.array(y_valid) - numpy.array(preds)), 2)))
print(rms)
# 计划
valid['Predictions'] = 0
valid['Predictions'] = preds

valid.index = new_data[987:].index
train.index = new_data[:987].index

pyplot.plot(train['Close'])
pyplot.plot(valid[['Close', 'Predictions']])
# 设置标题
pyplot.title("LinearRegression算法预测曲线", fontsize=24)
# 设置图例
pyplot.legend(["原始曲线", "真实曲线", "预测曲线"], fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()
"""
    线性回归完毕
    开始k近邻
    使用相同的数据集
"""

# k近邻
x_train_scaled = scaler.fit_transform(x_train)
x_train = pandas.DataFrame(x_train_scaled)
x_valid_scaled = scaler.fit_transform(x_valid)
x_valid = pandas.DataFrame(x_valid_scaled)

# 使用GridSearch查找最佳参数
params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9]}
knn = neighbors.KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)

# 拟合模型进行预测
model.fit(x_train, y_train)
preds = model.predict(x_valid)

# 使用rms验证
rms = numpy.sqrt(numpy.mean(numpy.power((numpy.array(y_valid) - numpy.array(preds)), 2)))
print(rms)

valid['Predictions'] = 0
valid['Predictions'] = preds
pyplot.plot(train['Close'])
pyplot.plot(valid[['Close', 'Predictions']])
# 设置标题
pyplot.title("KNN算法预测曲线", fontsize=24)
# 设置图例
pyplot.legend(["原始曲线", "真实曲线", "预测曲线"], fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()

# 使用ARIMA
data = df.sort_index(ascending=True, axis=0)

train = data[:987]
valid = data[987:]

training = train['Close']
validation = valid['Close']

model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                   trace=True, error_action='ignore', suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=248)
forecast = pandas.DataFrame(forecast, index=valid.index, columns=['Prediction'])

rms = numpy.sqrt(numpy.mean(numpy.power((numpy.array(valid['Close']) - numpy.array(forecast['Prediction'])), 2)))
print(rms)

pyplot.plot(train['Close'])
pyplot.plot(valid['Close'])
pyplot.plot(forecast['Prediction'])
# 设置标题
pyplot.title("ARIMA算法预测曲线", fontsize=24)
# 设置图例
pyplot.legend(["原始曲线", "真实曲线", "预测曲线"], fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()

# LSTM
data = df.sort_index(ascending=True, axis=0)

new_data = pandas.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# 设定指标
new_data.index = new_data.Date

new_data.drop('Date', axis=1, inplace=True)
dataset = new_data.values
train = dataset[0:987, :]
valid = dataset[987:, :]
scaled_data = scaler.fit_transform(dataset)
x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = numpy.array(x_train), numpy.array(y_train)
x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = numpy.array(X_test)
X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms = numpy.sqrt(numpy.mean(numpy.power((valid - closing_price), 2)))
print(rms)

train = new_data[:987]
valid = new_data[987:]
valid['Predictions'] = closing_price
pyplot.plot(train['Close'])
pyplot.plot(valid[['Close', 'Predictions']])
# 设置标题
pyplot.title("LSTM神经网络预测曲线", fontsize=24)
# 设置图例
pyplot.legend(["原始曲线", "真实曲线", "预测曲线"], fontsize=16)
# 设置x轴标签
pyplot.xlabel("日期", fontsize=24)
# 设置y轴标签
pyplot.ylabel("收盘价", fontsize=24)
# 设置x坐标轴字体
pyplot.xticks(fontsize=16)
# 设置y坐标轴字体
pyplot.yticks(fontsize=16)
pyplot.show()
