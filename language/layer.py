from functools import reduce

import numpy as np

from core.layer import Layer, Affine
from core.updater import Updater


class CBOW(Layer):
    _sub_layers: list[Affine]

    def __init__(self, W: np.ndarray, updater: Updater):
        self.params = [W]
        self.grads = [np.zeros(W.shape)]
        self._sub_layers = []
        self._updater = updater

    @classmethod
    def from_size(cls, input_size, hidden_size, updater: Updater, init_weight=0.01):
        W = np.random.randn(input_size, hidden_size) * init_weight
        return cls(W, updater)

    def forward(self, x: np.ndarray, **kwargs):
        context_size = x.shape[1]

        if len(self._sub_layers) == 0:
            for i in range(context_size):
                layer = Affine(self.params[0], None, self._updater, useBias=False)
                self._sub_layers.append(layer)

        out = np.zeros((x.shape[0], self.params[0].shape[1]))
        for i in range(context_size):
            layer = self._sub_layers[i]
            result = layer.forward(x[:, i])
            out += result

        return out / (context_size * 1.0)

    def backward(self, dout: np.ndarray):
        context_size = len(self._sub_layers)
        dout *= 1.0 / (context_size * 1.0)

        dx = []
        for layer in reversed(self._sub_layers):
            result = layer.backward(dout)
            dx.append(result)

        # sum all sub layer's gradients
        self.grads[0][...] = reduce(lambda x, y: x.grads[0] + y.grads[0], self._sub_layers[1:], self._sub_layers[0])

        return dx

    def update_params(self):
        self._updater.update(self.params, self.grads)


class Embedding(Layer):
    def __init__(self, W: np.ndarray, updater: Updater):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self._updater = updater
        self._x = None
        self._out_W = None

    def forward(self, x: np.ndarray, **kwargs):
        W = self.params[0]
        self._x = x
        out = W[self._x]
        self._out_W = out
        return out

    def backward(self, dout: np.ndarray):
        dW = self.grads[0]
        np.add.at(dW, self._x, dout)
        dx = np.sum(self._out_W * dout, axis=1)
        return dx

    def update_params(self):
        # This update_params method is only for single Embedding layer.
        self._updater.update(self.params, self.grads)


class MultiInputEmbedding(Layer):
    _sub_layers: list[Embedding]

    def __init__(self, W: np.ndarray, updater: Updater):
        self._sub_layers = []
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self._updater = updater
        self._x = None
        self._context_size = None

    @classmethod
    def from_size(cls, input_size, hidden_size, updater: Updater, init_weight=0.01):
        W = np.random.randn(input_size, hidden_size) * init_weight
        return cls(W, updater)

    def forward(self, x: np.ndarray, **kwargs):
        self._x = x
        self._context_size = x.shape[1]

        if len(self._sub_layers) == 0:
            for i in range(self._context_size):
                layer = Embedding(self.params[0], self._updater)
                self._sub_layers.append(layer)

        out = np.zeros((x.shape[0], self.params[0].shape[1]))

        for i in range(self._context_size):
            layer = self._sub_layers[i]
            result = layer.forward(x[:, i])
            out += result

        return out / (self._context_size * 1.0)

    def backward(self, dout: np.ndarray):
        # dout.shape: (m, W.shape[1])
        dout *= 1.0 / self._context_size
        dW = self.grads[0]

        dx = []
        for layer in self._sub_layers:
            result = layer.backward(dout)
            dx.append(result)

        # sum all sub layer's gradients
        dW[...] = reduce(lambda x, y: x.grads[0] + y.grads[0], self._sub_layers[1:], self._sub_layers[0])

        return dx

    def update_params(self):
        self.params = self._updater.update(self.params, self.grads)
        W = self.params[0]
        for layer in self._sub_layers:
            layer.params[0] = W
