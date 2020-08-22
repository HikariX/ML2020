import sys
import pandas as pd
import numpy as np
from dataPreprocessing import *
import matplotlib.pyplot as plt

def Problem1():
    X, y = Produce(time_window=9, selected_row=False)
    dim = 18 * 9 + 1
    x = np.concatenate((np.ones([12 * 471, 1]), X), axis=1).astype(float)
    iter_time = 5000
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    loss_dict = {}
    for learning_rate in [100, 10, 1, 0.1]:
        w = np.zeros([dim, 1])
        loss_list = []
        for t in range(iter_time):
            loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
            if t % 10 == 0:
                loss_list.append(loss)
            gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
            adagrad += gradient ** 2
            w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
        loss_dict[learning_rate] = loss_list
    xticks = [i for i in range(0, iter_time, 10)]
    plot1, = plt.plot(xticks, loss_dict[0.1], c='b')
    plot2, = plt.plot(xticks, loss_dict[1], c='g')
    plot3, = plt.plot(xticks, loss_dict[10], c='r')
    plot4, = plt.plot(xticks, loss_dict[100], c='y')
    plt.legend([plot1, plot2, plot3, plot4], (0.1, 1, 10, 100))
    plt.title('Different learning rate')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.show()

def Problem2():
    val_loss_dict = {}
    learning_rate = 10
    iter_time = 5000
    eps = 0.0000000001
    for time_window in [5, 9]:
        X, y = Produce(time_window=time_window, selected_row=False)
        X_train_set, y_train_set, X_validation, y_validation = dataSplit(X, y)
        dim = 18 * time_window + 1
        w = np.zeros([dim, 1])
        x_train = np.concatenate((np.ones([len(X_train_set), 1]), X_train_set), axis=1).astype(float)
        x_val = np.concatenate((np.ones([len(X_validation), 1]), X_validation), axis=1).astype(float)
        adagrad = np.zeros([dim, 1])
        val_loss_list = []
        for t in range(iter_time):
            loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2)) / len(X_train_set))  # rmse
            if t % 10 == 0:
                val_loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_validation, 2)) / len(X_validation))
                val_loss_list.append(val_loss)
            gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set)  # dim*1
            adagrad += gradient ** 2
            w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
        val_loss_dict[time_window] = val_loss_list
    xticks = [i for i in range(0, iter_time, 10)]
    plot1, = plt.plot(xticks, val_loss_dict[5], c='b')
    plot2, = plt.plot(xticks, val_loss_dict[9], c='g')
    plt.legend([plot1, plot2], ('5hr', '9hr'))
    plt.title('Different time window')
    plt.xlabel('Iter')
    plt.ylabel('Val Loss')
    plt.show()

def Problem3():
    val_loss_dict = {}
    learning_rate = 10
    iter_time = 5000
    eps = 0.0000000001
    for selected_row in [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [9]]:
        X, y = Produce(time_window=9, selected_row=selected_row)
        X_train_set, y_train_set, X_validation, y_validation = dataSplit(X, y)
        dim = len(selected_row) * 9 + 1
        w = np.zeros([dim, 1])
        x_train = np.concatenate((np.ones([len(X_train_set), 1]), X_train_set), axis=1).astype(float)
        x_val = np.concatenate((np.ones([len(X_validation), 1]), X_validation), axis=1).astype(float)
        adagrad = np.zeros([dim, 1])
        val_loss_list = []
        for t in range(iter_time):
            loss = np.sqrt(np.sum(np.power(np.dot(x_train, w) - y_train_set, 2)) / len(X_train_set))  # rmse
            if t % 10 == 0:
                val_loss = np.sqrt(np.sum(np.power(np.dot(x_val, w) - y_validation, 2)) / len(X_validation))
                val_loss_list.append(val_loss)
            gradient = 2 * np.dot(x_train.transpose(), np.dot(x_train, w) - y_train_set)  # dim*1
            adagrad += gradient ** 2
            w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
        if len(selected_row) == 18:
            val_loss_dict[0] = val_loss_list
        else:
            val_loss_dict[1] = val_loss_list
    xticks = [i for i in range(0, iter_time, 10)]
    plot1, = plt.plot(xticks, val_loss_dict[0], c='b')
    plot2, = plt.plot(xticks, val_loss_dict[1], c='g')
    plt.legend([plot1, plot2], ('All', ' Only PM2.5'))
    plt.title('Different feature taken')
    plt.xlabel('Iter')
    plt.ylabel('Val Loss')
    plt.show()

def Problem4():
    learning_rate = 10
    iter_time = 1000
    eps = 0.0000000001
    selected_row = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    time_window = 9
    X, y, X_square, mean_x, std_x, mean_x_square, std_x_square = Produce_square(time_window=time_window, selected_row=selected_row)
    dim = len(selected_row) * time_window + 1
    w1 = np.zeros([dim, 1])
    w2 = np.zeros([dim - 1, 1])
    X = np.concatenate((np.ones([len(X), 1]), X), axis=1).astype(float)
    adagrad1 = np.zeros([dim, 1])
    adagrad2 = np.zeros([dim - 1, 1])
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(X, w1) + np.dot(X_square, w2) - y, 2)) / len(
            X))  # rmse
        gradient1 = 2 * np.dot(X.transpose(),
                               np.dot(X, w1) + np.dot(X_square, w2) - y) / len(X)  # dim*1
        gradient2 = 2 * np.dot(X_square.transpose(),
                               np.dot(X, w1) + np.dot(X_square, w2) - y) / len(X)  # dim*1

        adagrad1 += gradient1 ** 2
        adagrad2 += gradient2 ** 2
        w1 = w1 - learning_rate * gradient1 / np.sqrt(adagrad1 + eps)
        w2 = w2 - learning_rate * gradient2 / np.sqrt(adagrad2 + eps)
    np.save('weight1.npy', w1)
    w1 = np.load('weight1.npy')
    np.save('weight2.npy', w2)
    w2 = np.load('weight2.npy')

    test_x, test_x_square = Produce_square_test(time_window, selected_row, mean_x, std_x, mean_x_square, std_x_square)
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)
    ans_y = np.dot(test_x, w1) + + np.dot(test_x_square, w2)
    import csv
    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            print(row)

# Problem1()
# Problem2()
# Problem3()
Problem4()
