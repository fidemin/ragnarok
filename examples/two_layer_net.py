import numpy as np
from keras.datasets import mnist


def convert_to_one_hot_encoding(y: np.ndarray):
    result = np.zeros((y.size, np.max(y) + 1))
    result[np.arange(y.size), y] = 1
    return result


if __name__ == '__main__':
    (train_X, train_y_origin), (test_X, test_y_origin) = mnist.load_data()
    train_y = convert_to_one_hot_encoding(train_y_origin)
    test_y = convert_to_one_hot_encoding(test_y_origin)

    train_loss_list = []

    iter_num = 10000
    train_size = train_X.size
    batch_size = 100
    learning_rate = 0.1

    print(train_y[:10])
    print(test_y[:10])
