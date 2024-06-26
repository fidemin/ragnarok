from functools import reduce

import numpy as np

from src.main.core import loss, activation
from src.main.core.activation import sigmoid, tanh
from src.main.core.layer import Layer, Affine, LayerException, SigmoidWithLoss, Softmax
from src.main.core.optimizer import Optimizer
from src.main.language.util import UnigramSampler


class CBOWInput(Layer):
    _sub_layers: list[Affine]

    def __init__(self, W: np.ndarray, updater: Optimizer):
        self.params = [W]
        self.grads = [np.zeros(W.shape)]
        self._sub_layers = []
        self._updater = updater

    @classmethod
    def from_size(cls, input_size, hidden_size, updater: Optimizer, init_weight=0.01):
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
        self._updater.optimize(self.params, self.grads)


class Embedding(Layer):
    def __init__(self, W: np.ndarray, updater: Optimizer = None):
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

        # Embedding layer has no output because it has no meaning.
        return None

    def update_params(self):
        # This update_params method is only for single Embedding layer.
        self._updater.optimize(self.params, self.grads)


class EmbeddingDot(Layer):
    indexes_key = 'indexes'

    def __init__(self, W: np.ndarray, updater: Optimizer):
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
        self._updater.optimize(self.params, self.grads)


class CBOWInputEmbedding(Layer):
    _sub_layers: list[Embedding]

    def __init__(self, W: np.ndarray, updater: Optimizer):
        self._sub_layers = []
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self._updater = updater
        self._x = None
        self._context_size = None

    @classmethod
    def from_size(cls, input_size, hidden_size, updater: Optimizer, init_weight=0.01):
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
        self._updater.optimize(self.params, self.grads)


class NegativeSampling(Layer):
    positive_indexes_key = 'positive_indexes'

    _embedding_dot_layers: list[EmbeddingDot]
    _loss_layers: list[SigmoidWithLoss]

    def __init__(self, W, negative_size: int, sampler: UnigramSampler, updater: Optimizer):
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
    def from_size(cls, input_size: int, words_size: int, negative_size: int, sampler: UnigramSampler,
                  updater: Optimizer,
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
        self._updater.optimize(self.params, self.grads)


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
        tanh_c_next = tanh(c_next)
        h_next = tanh_c_next * output_gate

        self._cache['tanh_c_next'] = tanh_c_next

        return h_next, c_next

    def backward(self, dh: np.ndarray, dc: np.ndarray):
        x = self._cache['x']
        h_prev = self._cache['h_prev']
        c_prev = self._cache['c_prev']
        tanh_c_next = self._cache['tanh_c_next']
        forget_gate = self._cache['forget_gate']
        remember_cell = self._cache['remember_cell']
        input_gate = self._cache['input_gate']
        output_gate = self._cache['output_gate']

        dtanh_h_next = dh * output_gate * (1 - np.square(tanh_c_next))
        ds = dc + dtanh_h_next

        dc_prev = ds * forget_gate
        doutput_gate = dh * tanh_c_next
        dZ_o = doutput_gate * output_gate * (1 - output_gate)

        dinput_gate = ds * remember_cell
        dZ_i = dinput_gate * input_gate * (1 - input_gate)

        dremember = ds * input_gate
        dZ_r = dremember * (1 - np.square(remember_cell))

        dforget_gate = ds * c_prev
        dZ_f = dforget_gate * forget_gate * (1 - forget_gate)

        # shape: N X 4H
        dZ = np.hstack((dZ_f, dZ_r, dZ_i, dZ_o))

        # dWx
        self.grads[0][...] = np.dot(x.T, dZ)
        # dWh
        self.grads[1][...] = np.dot(h_prev.T, dZ)
        # db
        self.grads[2][...] = np.sum(dZ, axis=0, keepdims=True)

        # shape: N * D
        dx = np.dot(dZ, self.params[0].T)
        # shape: N * H
        dh_prev = np.dot(dZ, self.params[1].T)

        return dx, dh_prev, dc_prev


class GroupedLSTM(Layer):
    def __init__(self, Wx: np.ndarray, Wh: np.ndarray, b: np.ndarray, stateful=False):

        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self._validate_params()
        self._stateful = stateful

        self._layers = None
        self._h = None
        self._dh = None
        self._c = None

    def _validate_params(self):
        Wx, Wh, b = self.params
        assert Wx.shape[1] == Wh.shape[1], "Wx.shape[1] == Wh.shape[1] is required"
        assert 4 * Wh.shape[0] == Wh.shape[1], "4 * Wh.shape[0] == Wh.shape[1] is required"
        assert 1 == b.shape[0], "b.shape[0] should be 1"
        assert Wh.shape[1] == b.shape[1], "Wh.shape[1] == b.shape[1] is required"

    def forward(self, xs: np.ndarray, **kwargs):
        Wx, Wh, b = self.params

        # N: batch size, T: subsequences size, D: input size
        N, T, D = xs.shape

        # H: hidden size
        H = Wh.shape[0]

        self._layers = []
        hs = np.zeros((N, T, H))

        if not self._stateful or self._h is None:
            self._h = np.zeros((N, H))

        if not self._stateful or self._c is None:
            self._c = np.zeros((N, H))

        for t in range(T):
            layer = LSTM(Wx, Wh, b)

            # save h, c as instance variable for forward propagation of next subsequences (truncated group)
            self._h, self._c = layer.forward(xs[:, t, :], self._h, self._c)
            hs[:, t, :] = self._h

            self._layers.append(layer)

        return hs

    def backward(self, dhs: np.ndarray):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        # initialize with zeros for truncated BPTT
        dh, dc = 0, 0
        grads = [0, 0, 0]

        dxs = np.zeros((N, T, D))

        for t in reversed(range(T)):
            layer = self._layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                # sum all gradients because same Wh, Wx, b is used in layers.
                grads[i] += grad

        self._dh = dh

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        return dxs

    def update_params(self):
        pass


class GroupedAffine(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray, updater: Optimizer = None):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self._updater = updater

        self._cache = {}

    def forward(self, xs: np.ndarray, **kwargs):
        N, T, D = xs.shape
        self._cache['original_shape'] = xs.shape
        W = self.params[0]
        b = self.params[1]

        H = W.shape[1]

        xs = xs.reshape(N * T, D)

        self._cache['xs'] = xs

        Z = np.dot(xs, W) + b

        return Z.reshape(N, T, H)

    def backward(self, dout: np.ndarray):
        N, T, H = dout.shape

        xs = self._cache['xs']
        original_shape = self._cache['original_shape']

        W = self.params[0]
        dW = self.grads[0]
        db = self.grads[1]

        # dW calculation
        dout = dout.reshape(N * T, H)
        dW[...] = np.dot(xs.T, dout)
        db[...] = np.sum(dout, axis=0)

        dxs = np.dot(dout, W.T)
        dxs = dxs.reshape(original_shape)
        return dxs

    def update_params(self):
        self._updater.optimize(self.params, self.grads)


class GroupedSoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self._cache = {}

    def forward(self, xs: np.ndarray, ts: np.ndarray):
        N, T, D = xs.shape
        self._cache['original_size'] = xs.shape
        xs = xs.reshape(N * T, D)
        ts = ts.reshape(N * T, D)

        ys = activation.softmax(xs)

        self._cache['ys'] = ys
        self._cache['ts'] = ts
        return loss.cross_entropy(ys, ts)

    def backward(self, dout=1):
        ys = self._cache['ys']
        ts = self._cache['ts']
        original_size = self._cache['original_size']

        batch_size = ys.shape[0]
        dxs = dout * (ys - ts) / batch_size

        return dxs.reshape(original_size)


class WeightSum(Layer):
    def __init__(self):
        self._cache = {}

    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        hs = inputs[0]  # N X T X H
        weight = inputs[1]  # N X T

        N, T, H = hs.shape

        weight = weight.reshape((N, T, 1)).repeat(H, axis=2)

        out = hs * weight
        out = out.sum(axis=1)

        self._cache['shape'] = hs.shape
        self._cache['hs'] = hs
        self._cache['weight'] = weight
        return out

    def backward(self, dout: np.ndarray):
        # dout: N X H
        N, T, H = self._cache['shape']
        hs = self._cache['hs']
        weight = self._cache['weight']

        dout = dout.reshape((N, 1, H)).repeat(T, axis=1)

        dhs = dout * weight
        dweight = dout * hs

        dweight = dweight.sum(axis=2)

        return dhs, dweight

    def update_params(self):
        pass


class WeightForAttention(Layer):
    def __init__(self):
        self._cache = {}

    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        hs_from_encoding = inputs[0]  # N X T X H
        h_from_decoding = inputs[1]  # N X H

        N, T, H = hs_from_encoding.shape

        h_from_decoding = h_from_decoding.reshape((N, 1, H)).repeat(T, axis=1)

        out = hs_from_encoding * h_from_decoding
        out = out.sum(axis=2)  # N X T

        softmax = Softmax()
        out = softmax.forward(out)

        self._cache['softmax'] = softmax
        self._cache['shape'] = hs_from_encoding.shape
        self._cache['hs_from_encoding'] = hs_from_encoding
        self._cache['h_from_decoding'] = h_from_decoding
        return out

    def backward(self, dout: np.ndarray):
        # dout: N X T

        softmax = self._cache['softmax']
        N, T, H = self._cache['shape']
        hs_from_encoding = self._cache['hs_from_encoding']
        h_from_decoding = self._cache['h_from_decoding']

        dout = softmax.backward(dout)

        dout = dout.reshape(N, T, 1).repeat(H, axis=2)

        dhs_from_encoding = dout * h_from_decoding
        dh_from_decoding = dout * hs_from_encoding

        dh_from_decoding = dh_from_decoding.sum(axis=1)

        return dhs_from_encoding, dh_from_decoding

    def update_params(self):
        pass


class Attention(Layer):
    def __init__(self):
        self._weight_sum_layer = WeightSum()
        self._weight_for_attention_layer = WeightForAttention()

    def forward(self, *inputs: np.ndarray, **kwargs):
        hs = inputs[0]
        h = inputs[1]

        weight = self._weight_for_attention_layer.forward(hs, h)
        out = self._weight_sum_layer.forward(hs, weight)
        return out

    def backward(self, dout: np.ndarray):
        dhs1, dweight = self._weight_sum_layer.backward(dout)
        dhs2, dh = self._weight_for_attention_layer.backward(dweight)
        dhs = dhs1 + dhs2
        return dhs, dh

    def update_params(self):
        pass


class GroupedAttention(Layer):
    def __init__(self):
        self._layers = None

    def forward(self, *inputs: np.ndarray, **kwargs):
        hs_enc = inputs[0]
        hs_dec = inputs[1]

        self._layers = []

        N, T_dec, H = hs_dec.shape

        out = np.zeros_like(hs_dec)

        for t in range(T_dec):
            attention = Attention()
            t_out = attention.forward(hs_enc, hs_dec[:, t, :])
            self._layers.append(attention)
            out[:, t, :] = t_out

        return out

    def backward(self, douts: np.ndarray):
        N, T_dec, H = douts.shape
        dhs_enc = 0
        dhs_dec = np.zeros_like(douts)

        for t in range(T_dec):
            dhs_enc_t, dhs_dec_t = self._layers[t].backward(douts[:, t, :])
            dhs_enc += dhs_enc_t
            dhs_dec[:, t, :] = dhs_dec_t

        return dhs_enc, dhs_dec

    def update_params(self):
        pass
