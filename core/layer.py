import numpy as np

from core import numerical


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
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

    def predict(self, x, activation_func):
        a = np.dot(x, self.W) + self.b
        return activation_func(a)

    def gradient(self, loss_func):
        self.W_grad = numerical.gradient(loss_func, self.W)
        self.b_grad = numerical.gradient(loss_func, self.b)

    def update_params_from_gradient_descent(self, lr=0.01):
        self.W -= self.W_grad * lr
        self.b -= self.b_grad * lr
