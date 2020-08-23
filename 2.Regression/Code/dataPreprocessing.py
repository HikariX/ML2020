import sys
import pandas as pd
import numpy as np

def getCleanedData(path, train=True):
    if train:
        data = pd.read_csv(path, encoding='big5')

        # 数据预处理
        data_new = data.iloc[:, 3:].copy(deep=True)  # 去除日期、站点、观测项目信息
    else:
        data = pd.read_csv(path, header=None, encoding='big5')
        data_new = data.iloc[:, 2:]

    data_new[data_new == 'NR'] = 0  # No Record 数据置0
    data_new = data_new.astype('float')
    raw_data = data_new.to_numpy()
    return raw_data

# 数据分组
# 数据DF的组合方式是行：测量项-天数，为18 * (20 * 12), 列：小时，为24。时间维度不存在同一方向，因此需要重新选取。
# 每个月的数据都会有天数断层，所以需要按照月份分开
def grouping(raw_data):
    month_data = {}
    for month in range(12):
        sample = np.empty([480, 18])
        for day in range(20):
            sample[day * 24:(day + 1) * 24, :] = raw_data[18 * (20 * month + day):18 * (20 * month + day + 1), :].T
        month_data[month] = sample
    return month_data

# 制作数据集
# 对于一个月内连续的20天，将它们的时间维度拼在一起，并滑动取数据。
def makeDataset(month_data, time_window, selected_row):
    start = 9 - time_window
    X = np.empty([12 * 471, len(selected_row) * time_window], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:  # 抵达最后一天的最后一块可分数据
                    continue
                X[month * 471 + day * 24 + hour, :] = month_data[month][day * 24 + hour + start:day * 24 + hour + 9,
                                                      selected_row].reshape(-1, )  # 将所有特征拼成行向量
                y[month * 471 + day * 24 + hour, 0] = month_data[month][day * 24 + hour + 9, 9]
    X = np.array(X)
    y = np.array(y)
    return X, y

def makeDataset_square(month_data, time_window, selected_row):
    start = 9 - time_window
    X = np.empty([12 * 471, len(selected_row) * time_window], dtype=float)
    X_square = np.empty([12 * 471, len(selected_row) * time_window], dtype=float)
    y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:  # 抵达最后一天的最后一块可分数据
                    continue
                X[month * 471 + day * 24 + hour, :] = month_data[month][day * 24 + hour + start:day * 24 + hour + 9,
                                                      selected_row].reshape(-1, )  # 将所有特征拼成行向量
                X_square[month * 471 + day * 24 + hour, :] = np.power(
                    month_data[month][day * 24 + hour + start:day * 24 + hour + 9,
                    selected_row].reshape(-1, ), 2)
                y[month * 471 + day * 24 + hour, 0] = month_data[month][day * 24 + hour + 9, 9]

    X = np.array(X)
    X_square = np.array(X_square)
    y = np.array(y)
    return X, y, X_square

# 进行正则化
def Normalization(X, mean_x=[], std_x=[]):
    if len(mean_x) == 0:
        mean_x = np.mean(X, axis=0)  # 18 * 9
        std_x = np.std(X, axis=0)  # 18 * 9
    for i in range(len(X)):  # 12 * 471
        for j in range(len(X[0])):  # 18 * 9
            if std_x[j] != 0:
                X[i][j] = (X[i][j] - mean_x[j]) / std_x[j]
    # 此处将mean和std传出是为了让测试数据使用训练集的分布进行归一化，否则会和模型设定不符。
    return X, mean_x, std_x

# 划分训练测试集
def dataSplit(X, y):
    import math
    X_train_set = X[: math.floor(len(X) * 0.8), :]
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    X_validation = X[math.floor(len(X) * 0.8):, :]
    y_validation = y[math.floor(len(y) * 0.8):, :]
    return X_train_set, y_train_set, X_validation, y_validation

# 产生数据
def Produce(input_path, time_window, selected_row):
    raw_data = getCleanedData(input_path)
    month_data = grouping(raw_data)
    X, y = makeDataset(month_data, time_window=time_window, selected_row=selected_row)
    X, mean_x, std_x = Normalization(X)
    return X, y, mean_x, std_x

# 产生含有平方项的数据
def Produce_square(input_path, time_window, selected_row):
    raw_data = getCleanedData(input_path)
    month_data = grouping(raw_data)
    X, y, X_square = makeDataset_square(month_data, time_window, selected_row)
    X, mean_x, std_x = Normalization(X)
    X_square, mean_x_square, std_x_square = Normalization(X_square)
    return X, y, X_square, mean_x, std_x, mean_x_square, std_x_square

# 对测试数据进行处理
def makeDataset_test(data, time_window, selected_row, isSquare):
    start = 9 - time_window
    X = np.empty([240, len(selected_row) * time_window], dtype=float)
    if isSquare:
        X_square = np.empty([240, len(selected_row) * time_window], dtype=float)
        for i in range(240):
            X[i, :] = data[18 * i:18 * (i + 1), start:9].T[:, selected_row].reshape(-1, )  # 将所有特征拼成行向量
            X_square[i, :] = np.power(data[18 * i:18 * (i + 1), start:9].T[:, selected_row].reshape(-1, ), 2)
        X = np.array(X)
        X_square = np.array(X_square)
        return X, X_square
    else:
        for i in range(240):
            X[i, :] = data[18 * i:18 * (i + 1), start:9].T[:, selected_row].reshape(-1, )  # 将所有特征拼成行向量
        X = np.array(X)
        return X

# 产生测试数据
def Produce_test(input_path, time_window, selected_row, mean_x, std_x):
    raw_data = getCleanedData(input_path, False)
    X = makeDataset_test(raw_data, time_window, selected_row, False)
    X, _, _ = Normalization(X, mean_x, std_x)
    return X

# 产生带有平方项的测试数据
def Produce_square_test(input_path, time_window, selected_row, mean_x, std_x, mean_x_square, std_x_square):
    raw_data = getCleanedData(input_path, False)
    X, X_square = makeDataset_test(raw_data, time_window, selected_row, True)
    X, _, _ = Normalization(X, mean_x, std_x)
    X_square, _, _ = Normalization(X_square, mean_x_square, std_x_square)
    return X, X_square

# 对含有平方项的数据进行划分
def dataSplit_square(X, y, X_square):
    import math
    X_train_set = X[: math.floor(len(X) * 0.8), :]
    X_validation = X[math.floor(len(X) * 0.8):, :]
    y_train_set = y[: math.floor(len(y) * 0.8), :]
    X_square_train_set = X_square[: math.floor(len(X_square) * 0.8), :]
    X_square_validation = X_square[math.floor(len(X_square) * 0.8):, :]
    y_validation = y[math.floor(len(y) * 0.8):, :]
    return X_train_set, y_train_set, X_validation, y_validation, X_square_train_set, X_square_validation

