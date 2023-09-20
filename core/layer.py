import copy

import numpy as np

from core import activation, loss


class Multiply:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return self.x * self.y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class Add:
    def forward(self, x, y):
        return x, y

    def backward(self, dout):
        return dout, dout


class ReLu:
    def __init__(self):
        self.x = None
        self.mask = None

    def forward(self, x: np.ndarray):
        self.mask = self.x <= 0
        out = copy.deepcopy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.x = None
        self.out = None

    def forward(self, x: np.ndarray):
        self.out = activation.sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)


class Affine:
    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, x: np.ndarray):
        self.x = x
        return np.dot(self.x, self.W) + self.b

    def backward(self, dout: np.ndarray):
        self.dW = np.dot(self.x.T)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx


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
