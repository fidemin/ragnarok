import copy
from abc import ABCMeta

import numpy as np

from core import activation, loss


class Layer(metaclass=ABCMeta):
    def forward(self, x: np.ndarray):
        raise NotImplementedError("Layer.forward should be implemented")

    def backward(self, dout: np.ndarray):
        raise NotImplementedError("Layer.backward should be implemented")

    def update_params_from_gradient(self, lr=0.01):
        raise NotImplementedError("Layer.update_params_from_gradient should be implemented")


class Relu(Layer):
    def __init__(self):
        self.x = None
        self.mask = None

    def forward(self, x: np.ndarray):
        self.x = x
        self.mask = self.x <= 0
        out = copy.deepcopy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

    def update_params_from_gradient(self, lr=0.01):
        pass


class Sigmoid(Layer):
    def __init__(self):
        self.x = None
        self.out = None

    def forward(self, x: np.ndarray):
        self.out = activation.sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

    def update_params_from_gradient(self, lr=0.01):
        pass


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

    @classmethod
    def from_sizes(cls, input_size: int, output_size: int, weight_init=0.01):
        W = weight_init * np.random.randn(input_size, output_size)
        b = np.zeros(output_size)
        return cls(W, b)

    def forward(self, x: np.ndarray):
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout: np.ndarray):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

    def update_params_from_gradient(self, lr=0.01):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = activation.softmax(x)
        self.t = t
        return loss.cross_entropy(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        return dout * (self.y - self.t) / batch_size
