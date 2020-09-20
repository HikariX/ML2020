import numpy as np
import matplotlib.pyplot as plt
from UsefulFunctions import _normalize, _train_dev_split, _shuffle, _f, _cross_entropy_loss, _predict, _accuracy, _gradient, _cross_entropy_loss_re, _gradient_re
np.random.seed(0)

def LogisticRegression_re(_lambda, X_train_fpath, Y_train_fpath, X_test_fpath, output_fpath):
    # Parse csv files to numpy array
    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

    # Normalize training and testing data
    X_train, X_mean, X_std = _normalize(X_train, train=True)
    X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

    # Split data into training set and development set
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]

    # Zero initialization for weights ans bias
    w = np.zeros((data_dim,))
    b = np.zeros((1,))

    # Some parameters for training
    max_iter = 10
    batch_size = 8
    learning_rate = 0.2

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # Calcuate the number of parameter updates
    step = 1

    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

            # Compute the gradient
            # w_grad, b_grad = _gradient(X, Y, w, b)
            w_grad, b_grad = _gradient_re(X, Y, w, b, _lambda)

            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        # train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
        train_loss.append(_cross_entropy_loss_re(y_train_pred, Y_train, w, _lambda) / train_size)

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss_re(y_dev_pred, Y_dev, w, _lambda) / dev_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()

    # Predict testing labels
    predictions = _predict(X_test, w, b)
    with open(output_fpath.format('logistic_re'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))

def LogisticRegression(norm, X_train_fpath, Y_train_fpath, X_test_fpath, output_fpath, lr, batch_size, iter):
    np.random.seed(0)

    # Parse csv files to numpy array
    with open(X_train_fpath) as f:
        next(f)
        X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
    with open(Y_train_fpath) as f:
        next(f)
        Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
    with open(X_test_fpath) as f:
        next(f)
        X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

    # # Normalize training and testing data
    if norm:
        X_train, X_mean, X_std = _normalize(X_train, train=True)
        X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)

    # Split data into training set and development set
    dev_ratio = 0.1
    X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]
    test_size = X_test.shape[0]
    data_dim = X_train.shape[1]

    # Zero initialization for weights ans bias
    w = np.zeros((data_dim,))
    b = np.zeros((1,))

    # Some parameters for training
    max_iter = iter
    batch_size = batch_size
    learning_rate = lr

    # Keep the loss and accuracy at every iteration for plotting
    train_loss = []
    dev_loss = []
    train_acc = []
    dev_acc = []

    # Calcuate the number of parameter updates
    step = 1

    # Iterative training
    for epoch in range(max_iter):
        # Random shuffle at the begging of each epoch
        X_train, Y_train = _shuffle(X_train, Y_train)

        # Mini-batch training
        for idx in range(int(np.floor(train_size / batch_size))):
            X = X_train[idx * batch_size:(idx + 1) * batch_size]
            Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

            # Compute the gradient
            # w_grad, b_grad = _gradient(X, Y, w, b)
            w_grad, b_grad = _gradient(X, Y, w, b)

            # gradient descent update
            # learning rate decay with time
            w = w - learning_rate / np.sqrt(step) * w_grad
            b = b - learning_rate / np.sqrt(step) * b_grad

            step = step + 1

        # Compute loss and accuracy of training set and development set
        y_train_pred = _f(X_train, w, b)
        Y_train_pred = np.round(y_train_pred)
        train_acc.append(_accuracy(Y_train_pred, Y_train))
        # train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)
        train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

        y_dev_pred = _f(X_dev, w, b)
        Y_dev_pred = np.round(y_dev_pred)
        dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
        dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

    print('Training loss: {}'.format(train_loss[-1]))
    print('Development loss: {}'.format(dev_loss[-1]))
    print('Training accuracy: {}'.format(train_acc[-1]))
    print('Development accuracy: {}'.format(dev_acc[-1]))

    # Loss curve
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    if norm:
        plt.savefig('./loss_normalized.png')
    else:
        plt.savefig('./loss.png')
    plt.show()

    # Accuracy curve
    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    if norm:
        plt.savefig('./acc_normalized.png')
    else:
        plt.savefig('acc.png')
    plt.show()

    # Predict testing labels
    predictions = _predict(X_test, w, b)
    with open(output_fpath.format('logistic'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))



import sys

LogisticRegression(True, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], 0.2, 8, 10)
