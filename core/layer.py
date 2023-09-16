import numpy as np


class Layer:
    def __init__(self, input_size, output_size, weight_init=0.01):
        """
        The layer object of neural network

        :param input_size: size of input for the layer
        :param output_size: size of output for the layer
        :param weight_init: initialization variable
        """

        self.W = weight_init * np.random.randn(input_size, output_size)
        self.b = np.zeros(output_size)

    def predict(self, x, activation_func):
        a = np.dot(x, self.W) + self.b
        return activation_func(a)
