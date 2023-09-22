import copy
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from core import activation, loss
from core.updater import SGD, Updater


class Layer(metaclass=ABCMeta):
    @property
    def updater(self) -> Optional[Updater]:
        if hasattr(self, '_updater'):
            return self._updater

        return None

    @updater.setter
    def updater(self, updater: Updater):
        if hasattr(self, '_updater'):
            self._updater = updater

    @property
    def grads(self) -> Optional[list[np.ndarray]]:
        if hasattr(self, '_grads'):
            return self._grads

        return None

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray):
        pass

    @abstractmethod
    def has_params(self):
        pass

    @abstractmethod
    def update_params(self):
        pass


class Relu(Layer):
    def __init__(self):
        self.x = None
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        self.mask = self.x <= 0
        out = copy.deepcopy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

    def has_params(self):
        return False

    def update_params(self):
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

    def has_params(self):
        return False

    def update_params(self, lr=0.01):
        pass


class Affine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.x = None

        # default updater
        self._updater = SGD()

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

    def has_params(self):
        return True

    def update_params(self):
        params = self._updater.update([self.W, self.b], [self.dW, self.db])
        self.W = params[0]
        self.b = params[1]


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
