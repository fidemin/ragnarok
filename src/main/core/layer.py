import copy
from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np

from src.main.core import loss, activation
from src.main.core.optimizer import Optimizer
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter
from src.main.ragnarok.nn.layer.activation import Sigmoid as SigmoidProxy
from src.main.ragnarok.nn.layer.linear import Linear as LinearProxy


class Layer(metaclass=ABCMeta):
    train_flag_key = "train_flag"

    params: list[np.ndarray]
    grads: list[np.ndarray]

    @abstractmethod
    def forward(self, x: np.ndarray, **kwargs):
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray):
        pass

    @abstractmethod
    def update_params(self):
        pass


class Relu(Layer):
    def __init__(self):
        self.x = None
        self.mask = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        self.x = x
        self.mask = self.x <= 0
        out = copy.deepcopy(x)
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

    def update_params(self):
        pass


class Sigmoid(Layer):
    _in_var: Optional[Variable]
    _out_var: Optional[Variable]

    def __init__(self):
        self.params = []
        self.grads = []
        self._proxy_layer = SigmoidProxy()
        self.x = None
        self._in_var = None
        self._out_var = None
        self.out = None

    def forward(self, x: np.ndarray, **kwargs):
        x_var = Variable(x)
        y_var = self._proxy_layer.forward(x_var)
        self.out = y_var.data
        self._in_var = x_var
        self._out_var = y_var
        return y_var.data

    def backward(self, dout):
        self._out_var.grad = Variable(dout)
        self._out_var.backward(retain_grad=True)
        return self._in_var.grad.data

    def update_params(self, lr=0.01):
        pass


class Affine(Layer):
    _in_var: Optional[Variable]
    _out_var: Optional[Variable]

    def __init__(
        self, W: np.ndarray, b: np.ndarray, updater: Optimizer = None, useBias=True
    ):
        if useBias:
            self.params = [W, b]
            self.grads = [np.zeros_like(W), np.zeros_like(b)]
        else:
            self.params = [W]
            self.grads = [np.zeros_like(W)]

        self._useBias = useBias

        self.x = None
        self._original_shape = None

        # default updater
        self._updater = updater

        self._proxy_layer = LinearProxy(
            in_size=W.shape[0], out_size=W.shape[1], use_bias=useBias
        )
        self._proxy_layer.params["W"] = Parameter(W)
        if useBias:
            self._proxy_layer.params["b"] = Parameter(b)
        self._in_var = None
        self._out_var = None

    @classmethod
    def from_sizes(
        cls,
        input_size: int,
        output_size: int,
        updater: Optimizer = None,
        init_weight=0.01,
        useBias=True,
    ):
        W = init_weight * np.random.randn(input_size, output_size)
        b = np.zeros(output_size)
        return cls(W, b, updater, useBias=useBias)

    def forward(self, x: np.ndarray, **kwargs):
        self._original_shape = x.shape
        # To handle tensor
        x = x.reshape(x.shape[0], -1)
        var_x = Variable(x)
        self.x = x
        y_var = self._proxy_layer.forward(var_x)

        self._in_var = var_x
        self._out_var = y_var

        return y_var.data

    def backward(self, dout: np.ndarray):
        self._out_var.grad = Variable(dout)

        self._out_var.backward(retain_grad=True)

        self.grads[0][...] = self._proxy_layer.params["W"].grad.data
        if self._useBias:
            self.grads[1][...] = self._proxy_layer.params["b"].grad.data

        dx = self._in_var.grad.data

        # to handle tensor
        dx = dx.reshape(self._original_shape)
        return dx

    def update_params(self):
        self._updater.optimize(self.params, self.grads)


class BatchNorm(Layer):
    def __init__(self, gamma, beta, optimizer: Optimizer = None):
        self.params = [gamma, beta]
        self.grads = [np.zeros_like(gamma), np.zeros_like(beta)]
        self._optimizer = optimizer

        self._eps = 1e-8
        self._x_norm = None
        self._xmu = None
        self._istd = None
        self._std = None
        self._var = None

    @classmethod
    def from_shape(cls, input_shape, optimizer: Optimizer = None):
        gamma = np.ones(input_shape)
        beta = np.zeros(input_shape)
        return cls(gamma, beta, optimizer)

    def forward(self, x: np.ndarray, **kwargs):
        gamma = self.params[0]  # size: (D, )
        beta = self.params[1]  # size: (D, )

        batch_size = x.shape[0] * 1.0
        avg = np.sum(x, axis=0) / batch_size
        xmu = x - avg
        self._xmu = xmu
        var = np.sum(np.power(xmu, 2), axis=0) / batch_size
        self._var = var

        e = 1e-8
        self._std = np.sqrt(var + e)  # size: (D, )
        self._istd = 1.0 / self._std  # size: (D, )

        x_norm = xmu * self._istd
        self._x_norm = x_norm

        return gamma * x_norm + beta

    def backward(self, dout: np.ndarray):
        N = dout.shape[0] * 1.0
        dgamma = self.grads[0]
        dbeta = self.grads[1]

        dgammax = dout  # same as size of x: (N, D)
        dbeta[...] = np.sum(dout, axis=0)  # size: (D, )

        dgamma[...] = np.sum(dgammax * self._x_norm, axis=0)  # size: (D, )
        dx_norm = dgammax * dgamma  # size: (N, D)

        distd = np.sum(dx_norm * self._xmu, axis=0)  # size: (D, )
        dxmu1 = dx_norm * self._istd  # size: (N, D)

        dstd = -1.0 / np.power(self._istd, 2) * distd  # size: (D, )

        dvar = 0.5 / (self._var + self._eps) * dstd  # size:(D, )
        dsq = np.ones(dout.shape) / N * dvar  # size: (N, D)

        dxmu2 = 2.0 * self._xmu * dsq  # size: (N, D)

        dxmu = dxmu1 + dxmu2  # size: (N, D)

        dx1 = 1.0 * dxmu  # size: (N, D)

        dmu = -np.sum(dxmu, axis=0)  # size: (D, )

        dx2 = np.ones(dout.shape) / N * dmu  # size: (N, D)

        dx = dx1 + dx2  # size: (N, D)

        return dx

    def update_params(self):
        pass


class Dropout(Layer):
    def __init__(self, dropout_ratio=0.5):
        self.params = []
        self.grads = []
        self._dropout_ratio = dropout_ratio
        self._mask = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        train_flag = True
        if self.train_flag_key in kwargs and kwargs["train_flag"] is False:
            train_flag = False

        if train_flag:
            self._mask = (np.random.rand(*x.shape) > self._dropout_ratio).astype(int)
            keep_ratio = 1.0 - self._dropout_ratio
            # inverted dropout: divided by keep_ratio to preserve expectation of x
            return x * self._mask / keep_ratio
        else:
            return x

    def backward(self, dout: np.ndarray):
        return dout * self._mask

    def update_params(self):
        pass


class Softmax(Layer):
    def __init__(self):
        self._cache = {}

    def forward(self, x: np.ndarray, **kwargs):
        out = activation.softmax(x)
        self._cache["out"] = out
        return out

    def backward(self, dout: np.ndarray):
        # derivative equation: dx_k = dout_k * out_k - out_k * sum_over_element(dout * out)
        # reference: https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax
        out = self._cache["out"]

        last_axis = len(out.shape) - 1

        dx = dout * out
        dsum = np.sum(dx, axis=last_axis, keepdims=True)

        dx = dx - out * dsum
        return dx

    def update_params(self):
        pass


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


class SigmoidWithLoss(Layer):
    t_key = "t"

    def __init__(self):
        self._t = None

        self._y = None

    def forward(self, x: np.ndarray, **kwargs):
        if self.t_key not in kwargs:
            raise LayerException("kwargs has the value with key t")

        t = kwargs[self.t_key]

        self._t = t

        self._y = activation.sigmoid(x)

        return loss.log_loss(self._y, self._t)

    def backward(self, dout=1):
        batch_size = self._y.shape[0]
        return dout * (self._y - self._t) / batch_size

    def update_params(self):
        pass


class LayerException(RuntimeError):
    pass
