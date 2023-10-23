from functools import reduce

import numpy as np

from core.activation import sigmoid, tanh
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
        dW[...] = 0
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
        self._updater.update(self.params, self.grads)


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

        negative_indexes = self._sampler.sample(batch_size, self._negative_size, exception_ids=positive_indexes)

        positive_kwargs = {EmbeddingDot.indexes_key: positive_indexes}
        out_positive = self._embedding_dot_layers[0].forward(x, **positive_kwargs)
        positive_labels = np.ones((batch_size, 1), dtype=np.int32)
        loss_kwargs = {SigmoidWithLoss.t_key: positive_labels}
        loss = self._loss_layers[0].forward(out_positive, **loss_kwargs)

        negative_labels = np.zeros((batch_size, 1), dtype=np.int32)
        for i in range(1, self._negative_size + 1):
            negative_kwargs = {EmbeddingDot.indexes_key: negative_indexes.T[i - 1]}
            out_negative = self._embedding_dot_layers[i].forward(x, **negative_kwargs)
            loss_kwargs = {SigmoidWithLoss.t_key: negative_labels}
            loss += self._loss_layers[i].forward(out_negative, **loss_kwargs)

        return loss

    def backward(self, dout: np.ndarray):
        dx = np.zeros(self._x.shape)
        for i in range(self._negative_size + 1):
            dloss = self._loss_layers[i].backward(dout)
            dx += self._embedding_dot_layers[i].backward(dloss)

        self.grads[0][...] = reduce(lambda grad, embedding_dot_layer: grad + embedding_dot_layer.grads[0],
                                    self._embedding_dot_layers[1:],
                                    self._embedding_dot_layers[0].grads[0])

        return dx

    def update_params(self):
        self._updater.update(self.params, self.grads)


class LSTM:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self._cache = {}

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        self._cache['x'] = x
        self._cache['h_prev'] = h_prev
        self._cache['c_prev'] = c_prev

        N, H = h_prev.shape

        Wx = self.params[0]
        Wh = self.params[1]
        b = self.params[2]
        Z = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b

        forget_gate = Z[:, :H]
        remember_cell = Z[:, H: 2 * H]
        input_gate = Z[:, 2 * H: 3 * H]
        output_gate = Z[:, 3 * H:]

        forget_gate = sigmoid(forget_gate)
        remember_cell = tanh(remember_cell)
        input_gate = sigmoid(input_gate)
        output_gate = sigmoid(output_gate)

        self._cache['forget_gate'] = forget_gate
        self._cache['remember_cell'] = remember_cell
        self._cache['input_gate'] = input_gate
        self._cache['output_gate'] = output_gate

        c_next = c_prev * forget_gate + remember_cell * input_gate
        h_next = tanh(c_next) * output_gate

        self._cache['h_next'] = h_next
        self._cache['c_next'] = c_next

        return h_next, c_next

    def backward(self, dh: np.ndarray, dc: np.ndarray):
        x = self._cache['x']
        h_prev = self._cache['h_prev']
        c_prev = self._cache['c_prev']
        forget_gate = self._cache['forget_gate']
        remember_cell = self._cache['remember_cell']
        input_gate = self._cache['input_gate']
        output_gate = self._cache['output_gate']

        dc_prev = dc * forget_gate

        doutput_gate = dh * output_gate * (1 - output_gate)

        dinput_gate_pre_sigmoid = dc * remember_cell
        dinput_gate = dinput_gate_pre_sigmoid * input_gate * (1 - input_gate)

        dremember_cell_pre_tanh = dc * input_gate
        dremember_cell = dremember_cell_pre_tanh * (1 - np.square(remember_cell))

        dforget = dc * c_prev

        # shape: N X 4H
        dZ = np.hstack((dforget, dremember_cell, dinput_gate, doutput_gate))

        self.grads[0][...] = np.dot(x.T, dZ)
        self.grads[1][...] = np.dot(h_prev.T, dZ)
        self.grads[2][...] = np.sum(dZ, axis=1, keepdims=True)

        # shape: N * D
        dx = np.dot(dZ, self.params[0].T)
        # shape: N * H
        dh_prev = np.dot(dZ, self.params[1].T)

        return dx, dh_prev, dc_prev
