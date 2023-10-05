import numpy as np
from keras.src.datasets import mnist


def convert_to_one_hot_encoding(y: np.ndarray):
    result = np.zeros((y.size, np.max(y) + 1))
    result[np.arange(y.size), y] = 1
    return result


def mnist_load_data(shuffle=False, reshape=True):
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    if reshape:
        train_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2])
    train_y = convert_to_one_hot_encoding(train_y)

    if reshape:
        test_X = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2])
    test_y = convert_to_one_hot_encoding(test_y)

    avg_X = np.average(train_X)
    std_X = np.std(train_X)

    train_X = (train_X - avg_X) / std_X
    test_X = (test_X - avg_X) / std_X

    if shuffle:
        size = train_X.shape[0]
        shuffle_idxs = np.arange(size)
        np.random.shuffle(shuffle_idxs)
        train_X = train_X[shuffle_idxs]
        train_y = train_y[shuffle_idxs]

    return (train_X, train_y), (test_X, test_y)
