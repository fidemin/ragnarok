from functools import reduce

import numpy as np

from core.layer import Layer, Affine, LayerException
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
        # x is indexes of words
        W = self.params[0]
        self._x = x
        out = W[self._x]
        self._out_W = out
        # out.shape = (x.shape[0], W.shape[1])
        return out

    def backward(self, dout: np.ndarray):
        dW = self.grads[0]
        np.add.at(dW, self._x, dout)

        # Embdding layer has no output because it has no meaning.
        return None

    def update_params(self):
        # This update_params method is only for single Embedding layer.
        self._updater.update(self.params, self.grads)


class EmbeddingDot(Layer):
    indexes_key = 'indexes'

    def __init__(self, W: np.ndarray, updater: Updater):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self._embedding_layer = Embedding(self.params[0].T, None)
        self._updater = updater
        self._indexes = None
        self._W_from_indexes = None
        self._x = None

    def forward(self, x: np.ndarray, **kwargs):
        # x.shape = (m, W.shape[0])
        self._x = x

        if self.indexes_key not in kwargs:
            raise LayerException('{} key is required as argument'.format(self.indexes_key))
        indexes = kwargs[self.indexes_key]
        self._indexes = indexes

        # W_from_indexes.shape = (m, W.shape[0])
        W_from_indexes = self._embedding_layer.forward(indexes)
        self._W_from_indexes = W_from_indexes
        out = np.sum(x * W_from_indexes, axis=1, keepdims=True)
        # out.shape = (m, 1)
        return out

    def backward(self, dout: np.ndarray):
        # Because sum is used in forward, use broadcasting
        self._embedding_layer.backward(self._x * dout)

        # Because sum is used in forward, use broadcasting
        dx = self._W_from_indexes * dout
        self.grads[0][...] = self._embedding_layer.grads[0].T

        # dx.shape = (m, W.shape[0])
        return dx

    def update_params(self):
        self._updater.update(self.params, self.grads)
        self._embedding_layer.params[0] = self.params[0].T


class CBOWInputEmbedding(Layer):
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
                # The updater for sublayer is not required
                layer = Embedding(self.params[0], None)
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

        for layer in self._sub_layers:
            layer.backward(dout)

        # sum all sub layer's gradients
        dW[...] = reduce(lambda x, y: x.grads[0] + y.grads[0], self._sub_layers[1:], self._sub_layers[0])

        # CBOW input has no dx because it has no meaning and not used.
        return None

    def update_params(self):
        self.params = self._updater.update(self.params, self.grads)
        W = self.params[0]
        for layer in self._sub_layers:
            layer.params[0] = W
