import numpy as np
from matplotlib import pyplot as plt

from core.layer import Affine, Sigmoid, Dropout
from core.net import Net
from core.updater import SGD
from examples.common import mnist_load_data

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist_load_data(shuffle=True)
    train_X = train_X[:300]
    train_y = train_y[:300]

    iter_num = 1000000000
    train_size = train_X.shape[0]
    batch_size = 100
    iter_per_epoch = 50
    learning_rate = 0.05

    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    hidden_size = 100

    layers = []

    # xavier init
    init_weight1 = 1.0 / np.sqrt(input_size)
    layer1 = Affine.from_sizes(input_size, hidden_size, SGD(lr=learning_rate), init_weight=init_weight1)
    layers.append(layer1)

    layer2 = Sigmoid()
    layers.append(layer2)

    dropout_layer = Dropout()
    layers.append(dropout_layer)

    # xavier init
    init_weight_mid = 1.0 / np.sqrt(hidden_size)

    for i in range(1):
        affine_layer = Affine.from_sizes(hidden_size, hidden_size, SGD(lr=learning_rate), init_weight=init_weight_mid)
        layers.append(affine_layer)
        activation_layer = Sigmoid()
        layers.append(activation_layer)
        dropout_layer = Dropout()
        layers.append(dropout_layer)

    final_affine_layer = Affine.from_sizes(hidden_size, output_size, SGD(lr=learning_rate), init_weight=init_weight_mid)
    layers.append(final_affine_layer)

    network = Net(layers)

    epoch_count = 0
    max_epoch = 100

    train_accuracies = []
    test_accuracies = []
    loss_list = []

    for i in range(iter_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_X[batch_mask]
        y_batch = train_y[batch_mask]

        network.gradient_descent(x_batch, y_batch)

        if i % iter_per_epoch == 0:
            epoch_count += 1
            train_accuracy = network.accuracy(train_X, train_y)
            test_accuracy = network.accuracy(test_X, test_y)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            loss = network.loss(train_X, train_y)
            loss_list.append(loss)
            print('iter_num: {}, epoch_count: {}, train_acc: {}, test_acc: {}, train_loss: {}'.format(i, epoch_count,
                                                                                                      train_accuracy,
                                                                                                      test_accuracy,
                                                                                                      loss))

            if epoch_count >= max_epoch:
                break

    x = list(range(1, len(train_accuracies) + 1))

    plt.subplot(2, 1, 1)
    plt.plot(x, train_accuracies, label='train')
    plt.plot(x, test_accuracies, label='test')
    plt.ylabel('accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, loss_list)
    plt.ylabel('loss')

    plt.show()
