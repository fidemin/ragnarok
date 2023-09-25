import numpy as np

from examples.common import mnist_load_data
from naive import net


def convert_to_one_hot_encoding(y: np.ndarray):
    result = np.zeros((y.size, np.max(y) + 1))
    result[np.arange(y.size), y] = 1
    return result


if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist_load_data()

    loss_list = []

    # iter_num = 10000
    iter_num = 10
    train_size = train_X.shape[0]
    batch_size = 100
    learning_rate = 0.1

    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    hidden_size = 50

    two_layer_net = net.Net([input_size, hidden_size, output_size])

    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_X[batch_mask]
        y_batch = train_y[batch_mask]

        two_layer_net.gradient_descent(x_batch, y_batch, lr=0.1)

        loss = two_layer_net.loss(x_batch, y_batch)
        loss_list.append(loss)

    print(loss_list)
