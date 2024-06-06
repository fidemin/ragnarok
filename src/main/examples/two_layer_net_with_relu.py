import matplotlib.pyplot as plt

from src.main.core import layer
from src.main.core import net
from src.main.examples.common import mnist_load_data

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist_load_data()

    loss_list = []

    iter_num = 2000
    train_size = train_X.shape[0]
    batch_size = 100
    learning_rate = 0.1

    input_size = train_X.shape[1]
    output_size = train_y.shape[1]
    hidden_size = 50

    # updater1 = SGD(lr=0.05)
    # updater1 = Momentum(lr=0.05, momentum=0.9)
    # updater1 = AdaGrad(lr=0.1)
    updater1 = Adam()
    # updater2 = SGD(lr=0.05)
    # updater2 = Momentum(lr=0.05, momentum=0.9)
    # updater2 = AdaGrad(lr=0.1)
    updater2 = Adam()

    # He init
    init_weight1 = 2 / np.sqrt(input_size)
    layer1 = layer.Affine.from_sizes(input_size, hidden_size, updater1, init_weight=init_weight1)
    layer2 = layer.Relu()
    init_weight2 = 2 / np.sqrt(hidden_size)
    layer3 = layer.Affine.from_sizes(hidden_size, output_size, updater2, init_weight=init_weight2)

    layers = [layer1, layer2, layer3]

    two_layer_net = net.Net(layers)

    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_X[batch_mask]
        y_batch = train_y[batch_mask]

        two_layer_net.gradient_descent(x_batch, y_batch)

        loss = two_layer_net.loss(x_batch, y_batch)
        loss_list.append(loss)

    # print(loss_list)
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()
