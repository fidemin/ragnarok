import numpy as np

from ragnarok.core.function.common import Function
from ragnarok.core.function.math import log
from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.activation import softmax


class MeanSquaredError(Function):
    def forward(self, *tensors: Tensor, **kwargs) -> Tensor:
        x0, x1 = tensors
        diff = x0 - x1
        length = len(diff) if diff.ndim > 0 else 1  # handle scalar
        return (diff**2).sum() / length

    def backward(self, *douts: Tensor):
        dout = douts[0]

        x0, x1 = self.inputs
        diff = x0 - x1

        length = len(diff) if diff.ndim > 0 else 1
        dx0 = 2 * diff * dout / length
        dx1 = -dx0

        return dx0, dx1

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 2:
            raise ValueError("MeanSquaredError requires 2 tensors")
        if tensors[0].shape != tensors[1].shape:
            raise ValueError("MeanSquaredError requires the same shape tensors")


class CrossEntropyError(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        y, t = tensors
        y_data = y.data
        t_data = t.data

        if y.ndim == 1:
            t_data = t_data.reshape(1, t_data.size)
            y_data = y_data.reshape(1, y_data.size)

        h = 1e-7
        batch_size = y_data.shape[0]

        return Tensor(-np.sum(t_data * np.log(y_data + h)) / batch_size)

    def backward(self, *douts: Tensor):
        dout = douts[0]
        y, t = self.inputs

        if y.ndim == 1:
            batch_size = 1
        else:
            batch_size = y.shape[0]

        h = 1e-7
        dx = dout * -(t / (y + h)) / batch_size
        dt = dout * -log(y + h) / batch_size

        return dx, dt

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 2:
            raise ValueError("CrossEntropyError requires 2 tensors")

        if tensors[0].shape != tensors[1].shape:
            raise ValueError("CrossEntropyError requires the same shape tensors")


def cross_entropy_error(y: Tensor, t: Tensor) -> Tensor:
    return CrossEntropyError()(y, t)


class SoftMaxLoss(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        y = softmax(tensors[0])
        self._cache["y"] = y
        t = tensors[1]
        return cross_entropy_error(y, t)

    def backward(self, *douts: Tensor):
        dout = douts[0]
        y = self._cache["y"]
        t = self.inputs[1]

        if y.ndim == 1:
            batch_size = 1
        else:
            batch_size = y.shape[0]

        dy = dout * (y - t) / batch_size
        dt = -dout * log(y) / batch_size
        return dy, dt

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 2:
            raise ValueError("SoftMaxLoss requires 2 tensors")
