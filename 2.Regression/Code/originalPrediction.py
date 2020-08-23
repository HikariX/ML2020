import sys
import pandas as pd
import numpy as np
from dataPreprocessing import *
import matplotlib.pyplot as plt

# 该代码完成最基础的预测功能。
def Prediction(input_path, output_path):
    learning_rate = 10
    iter_time = 1000
    eps = 0.0000000001
    selected_row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    time_window = 9
    X, y, mean_x, std_x = Produce('./Dataset/train.csv', time_window, selected_row)
    dim = len(selected_row) * time_window + 1
    w1 = np.zeros([dim, 1])
    X = np.concatenate((np.ones([len(X), 1]), X), axis=1).astype(float)
    adagrad1 = np.zeros([dim, 1])
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(X, w1) - y, 2)) / len(
            X))  # rmse
        gradient1 = 2 * np.dot(X.transpose(),
                               np.dot(X, w1) - y) / len(X)  # dim*1
        adagrad1 += gradient1 ** 2
        w1 = w1 - learning_rate * gradient1 / np.sqrt(adagrad1 + eps)
    np.save('original_weight1.npy', w1)
    w1 = np.load('original_weight1.npy')

    test_x = Produce_test(input_path, time_window, selected_row, mean_x, std_x)
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
    ans_y = np.dot(test_x, w1)
    import csv
    with open(output_path, mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)


Prediction(sys.argv[1], sys.argv[2])