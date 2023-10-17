from functools import reduce

import numpy as np

from core.layer import Layer, Affine, LayerException, SigmoidWithLoss
from core.updater import Updater
from language.util import UnigramSampler


class CBOWInput(Layer):
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
        self.grads[0][...] = reduce(lambda grad, sub_layer: grad + sub_layer.grads[0], self._sub_layers[1:],
                                    self._sub_layers[0].grads[0])

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
        dW[...] = reduce(lambda grad, sub_layer: grad + sub_layer.grads[0], self._sub_layers[1:],
                         self._sub_layers[0].grads[0])

        # CBOW input has no dx because it has no meaning and not used.
        return None

    def update_params(self):
        self.params = self._updater.update(self.params, self.grads)
        W = self.params[0]
        for layer in self._sub_layers:
            layer.params[0] = W


class NegativeSampling(Layer):
    positive_indexes_key = 'positive_indexes'

    _embedding_dot_layers: list[EmbeddingDot]
    _loss_layers: list[SigmoidWithLoss]

    def __init__(self, W, negative_size: int, sampler: UnigramSampler, updater: Updater):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self._sampler = sampler
        self._negative_size = negative_size
        self._updater = updater

        # positive layer 1 + negative layers
        # fist layer is for positive
        self._embedding_dot_layers = [EmbeddingDot(W, None) for _ in range(negative_size + 1)]
        self._loss_layers = [SigmoidWithLoss() for _ in range(negative_size + 1)]

        self._x = None

    @classmethod
    def from_size(cls, input_size: int, words_size: int, negative_size: int, sampler: UnigramSampler, updater: Updater,
                  init_weight=0.01):

        W = np.random.randn(input_size, words_size) * init_weight
        return cls(W, negative_size, sampler, updater)

    def forward(self, x: np.ndarray, **kwargs):
        self._x = x
        batch_size = x.shape[0]
        positive_indexes = kwargs[self.positive_indexes_key]

        negative_indexes = []
        for positive_idx in positive_indexes:
            negative_indexes.append(self._sampler.sample(self._negative_size, [positive_idx]))

        positive_kwargs = {EmbeddingDot.indexes_key: positive_indexes}
        out_positive = self._embedding_dot_layers[0].forward(x, **positive_kwargs)
        positive_labels = np.ones((batch_size, 1), dtype=np.int32)
        loss_kwargs = {SigmoidWithLoss.t_key: positive_labels}
        loss = self._loss_layers[0].forward(out_positive, **loss_kwargs)

        negative_labels = np.ones((batch_size, 1), dtype=np.int32)
        for i in range(1, self._negative_size + 1):
            negative_kwargs = {EmbeddingDot.indexes_key: [row[i - 1] for row in negative_indexes]}
            out_negative = self._embedding_dot_layers[i].forward(x, **negative_kwargs)
            loss_kwargs = {SigmoidWithLoss.t_key: negative_labels}
            loss += self._loss_layers[i].forward(out_negative, **loss_kwargs)

        return loss

    def backward(self, dout: np.ndarray):
        dx = np.zeros(self._x.shape)
        for i in reversed(range(1, self._negative_size + 1)):
            dloss_negative = self._loss_layers[i].backward(dout)
            dx += self._embedding_dot_layers[i].backward(dloss_negative)

        dloss_positive = self._loss_layers[0].backward(dout)
        dx += self._embedding_dot_layers[0].backward(dloss_positive)

        self.grads[0][...] = reduce(lambda grad, embedding_dot_layer: grad + embedding_dot_layer.grads[0],
                                    self._embedding_dot_layers[1:],
                                    self._embedding_dot_layers[0].grads[0])

        return dx

    def update_params(self):
        self._updater.update(self.params, self.grads)
