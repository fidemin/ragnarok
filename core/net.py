from core.activation import sigmoid
from core.layer import Layer


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


class NetInitException(Exception):
    pass
