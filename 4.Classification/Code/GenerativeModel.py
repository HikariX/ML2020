import numpy as np
import matplotlib.pyplot as plt
from UsefulFunctions import _normalize, _train_dev_split, _shuffle, _f, _cross_entropy_loss, _predict, _accuracy, _gradient

np.random.seed(0)
def GenerativeModel(norm, X_train_fpath, Y_train_fpath, X_test_fpath, output_fpath):
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
    if norm:
        X_train, X_mean, X_std = _normalize(X_train, train=True)
        X_test, _, _ = _normalize(X_test, train=False, specified_column=None, X_mean=X_mean, X_std=X_std)
    data_dim = X_train.shape[1]

    # Compute in-class mean
    X_train_0 = np.array([x for x, y in zip(X_train, Y_train) if y == 0])
    X_train_1 = np.array([x for x, y in zip(X_train, Y_train) if y == 1])

    mean_0 = np.mean(X_train_0, axis=0)
    mean_1 = np.mean(X_train_1, axis=0)

    # Compute in-class covariance
    cov_0 = np.zeros((data_dim, data_dim))
    cov_1 = np.zeros((data_dim, data_dim))

    for x in X_train_0:
        cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
    for x in X_train_1:
        cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

    # Shared covariance is taken as a weighted average of individual in-class covariance.
    cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

    # Compute inverse of covariance matrix.
    # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
    # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
    u, s, v = np.linalg.svd(cov, full_matrices=False)
    inv = np.matmul(v.T * 1 / s, u.T)

    # Directly compute weights and bias
    w = np.dot(inv, mean_0 - mean_1)
    b = (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1)) \
        + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

    # Compute accuracy on training set
    Y_train_pred = 1 - _predict(X_train, w, b)
    print('Training accuracy: {}'.format(_accuracy(Y_train_pred, Y_train)))

    # Predict testing labels
    predictions = 1 - _predict(X_test, w, b)
    with open(output_fpath.format('generative'), 'w') as f:
        f.write('id,label\n')
        for i, label in enumerate(predictions):
            f.write('{},{}\n'.format(i, label))


import sys

GenerativeModel(True, sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
