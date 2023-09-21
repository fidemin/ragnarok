import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

from core import net, layer


def convert_to_one_hot_encoding(y: np.ndarray):
    result = np.zeros((y.size, np.max(y) + 1))
    result[np.arange(y.size), y] = 1
    return result


if __name__ == '__main__':
    (train_X_origin, train_y_origin), (test_X_origin, test_y_origin) = mnist.load_data()
    train_X = train_X_origin.reshape(train_X_origin.shape[0], train_X_origin.shape[1] * train_X_origin.shape[2])
    train_y = convert_to_one_hot_encoding(train_y_origin)
    test_X = test_X_origin.reshape(test_X_origin.shape[0], test_X_origin.shape[1] * test_X_origin.shape[2])
    test_y = convert_to_one_hot_encoding(test_y_origin)

    loss_list = []

    iter_num = 5000
    train_size = train_X.shape[0]
    batch_size = 100
    learning_rate = 0.1

    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    hidden_size = 50

    layer1 = layer.Affine.from_sizes(input_size, hidden_size)
    layer2 = layer.Sigmoid()
    layer3 = layer.Affine.from_sizes(hidden_size, output_size)

    layers = [layer1, layer2, layer3]

    two_layer_net = net.Net(layers)

    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_X[batch_mask]
        y_batch = train_y[batch_mask]

        two_layer_net.gradient_descent(x_batch, y_batch, lr=0.05)

        loss = two_layer_net.loss(x_batch, y_batch)
        loss_list.append(loss)

    # print(loss_list)
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()
