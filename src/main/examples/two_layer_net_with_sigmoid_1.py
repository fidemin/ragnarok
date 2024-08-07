import numpy as np
from matplotlib import pyplot as plt

from src.main.core.layer import Affine, Sigmoid, SoftmaxWithLoss
from src.main.core.net import NeuralNet
from src.main.core.optimizer import SGD
from src.main.examples.common import mnist_load_data

if __name__ == "__main__":
    (origin_train_X, origin_train_y), (test_X, test_y) = mnist_load_data()

    train_size = origin_train_X.shape[0]
    mini_batch_size = 100
    iter_num = train_size // mini_batch_size
    max_epoch = 20

    input_size = origin_train_X.shape[1]
    output_size = origin_train_y.shape[1]
    hidden_size = 50

    # xavier init
    init_weight1 = 1 / np.sqrt(input_size)
    layer1 = Affine.from_sizes(input_size, hidden_size, init_weight=init_weight1)
    layer2 = Sigmoid()
    init_weight2 = 1 / np.sqrt(hidden_size)
    layer3 = Affine.from_sizes(hidden_size, output_size, init_weight=init_weight2)
    loss_layer = SoftmaxWithLoss()
    two_layer_net = NeuralNet([layer1, layer2, layer3], loss_layer, SGD(lr=0.01))

    loss_list = []
    accuracy_list = []

    for epoch in range(max_epoch):
        indexes = np.random.permutation(np.arange(train_size))
        train_X = origin_train_X[indexes]
        train_y = origin_train_y[indexes]

        for i in range(iter_num):
            mini_batch_X = train_X[mini_batch_size * i : mini_batch_size * (i + 1)]
            mini_batch_y = train_y[mini_batch_size * i : mini_batch_size * (i + 1)]

            loss = two_layer_net.forward(mini_batch_X, mini_batch_y)
            two_layer_net.backward()
            two_layer_net.optimize()

            loss_list.append(loss)

        accuracy = two_layer_net.accuracy(test_X, test_y)
        accuracy_list.append(accuracy)
        print("epoch: {}, loss: {}".format(epoch + 1, loss_list[len(loss_list) - 1]))

    plt.subplot(2, 1, 1)
    loss_x = list(range(1, len(loss_list) + 1))
    plt.plot(loss_x, loss_list)
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.subplot(2, 1, 2)
    accuracy_x = list(range(1, len(accuracy_list) + 1))
    plt.plot(accuracy_x, accuracy_list)
    plt.xlabel("epoch")
    plt.ylabel("accuracy by epoch")

    plt.show()
