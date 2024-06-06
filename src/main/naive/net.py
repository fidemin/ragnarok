import numpy as np

from src.main.core import loss
from src.main.core.activation import sigmoid, softmax
from src.main.naive.layer import Layer


class Net:
    def __init__(self, sizes: list, activation_func=sigmoid):
        """
        :param sizes: the size of each layer.
            [2, 3, 10] means the sizes of input layer, hidden layer, output layer are (2, 3, 10)
        :param activation_func: activation function for this neural network.
        """

        if len(sizes) < 2:
            raise NetInitException("The sizes should have at least two elements")

        self.layers = []
        self.activation_func = activation_func

        for i in range(len(sizes) - 1):
            layer = Layer(sizes[i], sizes[i + 1])
            self.layers.append(layer)

    def predict(self, x: np.ndarray) -> np.ndarray:
        size = len(self.layers)

        for i in range(size - 1):
            x = self.layers[i].predict(x, self.activation_func)

        return self.layers[size - 1].predict(x, softmax)

    def loss(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)

        return loss.cross_entropy(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)

        y_max_idx = np.argmax(y, axis=1)
        t_max_idx = np.argmax(t, axis=1)

        return np.sum(y_max_idx == t_max_idx) / float(x.shape[0])

    def gradient(self, x, t):
        loss_func = lambda W: self.loss(x, t)

        for i in range(len(self.layers)):
            self.layers[i].gradient(loss_func)

    def gradient_descent(self, x: np.ndarray, t: np.ndarray, lr=0.01):
        loss_func = lambda W: self.loss(x, t)

        # first calculate gradient for each layer with CURRENT W, b of layers.
        for i in range(len(self.layers)):
            self.layers[i].gradient(loss_func)

        # second update parameter W, b in layers.
        for i in range(len(self.layers)):
            self.layers[i].update_params_from_gradient_descent(lr)


class NetInitException(Exception):
    pass
