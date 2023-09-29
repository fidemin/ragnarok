import numpy as np
from keras.src.datasets import mnist


def convert_to_one_hot_encoding(y: np.ndarray):
    result = np.zeros((y.size, np.max(y) + 1))
    result[np.arange(y.size), y] = 1
    return result


def mnist_load_data(shuffle=False):
    (train_X_origin, train_y_origin), (test_X_origin, test_y_origin) = mnist.load_data()
    train_X_temp = train_X_origin.reshape(train_X_origin.shape[0], train_X_origin.shape[1] * train_X_origin.shape[2])
    train_y = convert_to_one_hot_encoding(train_y_origin)
    test_X_temp = test_X_origin.reshape(test_X_origin.shape[0], test_X_origin.shape[1] * test_X_origin.shape[2])
    test_y = convert_to_one_hot_encoding(test_y_origin)

    avg_X = np.average(train_X_temp)
    std_X = np.std(train_X_temp)

    train_X = (train_X_temp - avg_X) / std_X
    test_X = (test_X_temp - avg_X) / std_X

    if shuffle:
        size = train_X.shape[0]
        shuffle_idxs = np.arange(size)
        np.random.shuffle(shuffle_idxs)
        train_X = train_X[shuffle_idxs]
        train_y = train_y[shuffle_idxs]

    return (train_X, train_y), (test_X, test_y)
