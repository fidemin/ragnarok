import numpy as np
from matplotlib import pyplot as plt

from cnn.layer import Convolution, Pooling
from core.layer import Relu, Affine
from core.net import Net
from core.updater import SGD
from examples.common import mnist_load_data

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist_load_data(reshape=False)
    train_X = np.expand_dims(train_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    train_size = train_X.shape[0]
    C = 1  # image has no color
    W = train_X.shape[2]
    H = train_X.shape[3]

    # FN, FC, FH, FW
    FN = 30
    FC = 1
    FH = 5
    FW = 5
    F_shape = (FN, FC, FH, FW)

    stride = 1
    padding = 0
    OH = (H + 2 * padding - FH) // stride + 1
    OW = (W + 2 * padding - FW) // stride + 1

    init_weight1 = 2 / np.sqrt(C * W * H)

    # layer1.forward -> (N, FN, OH, OW)
    layer1 = Convolution.from_sizes(F_shape, SGD(), stride=stride, padding=padding, weight_init=init_weight1)

    layer2 = Relu()

    PH = 2
    PW = 2
    P_stride = 2
    P_padding = 0
    layer3 = Pooling(pool_h=PH, pool_w=PW, stride=P_stride, padding=P_padding)

    pooing_out_channel = FN
    pooling_out_h = (OH + 2 * P_padding - PH) // P_stride + 1
    pooling_out_w = (OW + 2 * P_padding - PW) // P_stride + 1

    affine_input_size = FN * pooling_out_h * pooling_out_w
    hidden_size = 100
    weight_init2 = 2 / np.sqrt(affine_input_size)
    layer4 = Affine.from_sizes(affine_input_size, hidden_size, SGD(), init_weight=weight_init2)

    layer5 = Relu()

    output_size = 10
    weight_init3 = 2 / np.sqrt(output_size)
    layer6 = Affine.from_sizes(input_size=hidden_size, output_size=output_size, updater=SGD(), init_weight=weight_init3)

    net = Net([layer1, layer2, layer3, layer4, layer5, layer6])

    loss_list = []
    iter_num = 2000
    batch_size = 100
    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = train_X[batch_mask]
        y_batch = train_y[batch_mask]

        net.gradient_descent(x_batch, y_batch)

        loss = net.loss(x_batch, y_batch)
        loss_list.append(loss)

        # print(loss_list)
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()
