import numpy as np

from ragnarok.core.function.common import Function, FunctionVariableError
from ragnarok.core.function.math import exp
from ragnarok.core.tensor import Tensor


class Sigmoid(Function):
    def forward(self, *variables: Tensor, **kwargs):
        x = variables[0]
        return 1 / (1 + exp(-x))

    def backward(self, *douts: Tensor):
        dout = douts[0]
        y = self._outputs()[0]
        return dout * y * (1.0 - y)

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if len(variables) != 1:
            raise ValueError("Sigmoid requires 1 tensor")


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid()(x)


class Tanh(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(np.tanh(x.data))

    def backward(self, *douts: Tensor):
        dout = douts[0]
        y = self._outputs()[0]
        return (1 - y**2) * dout

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Tanh function."
            )


def tanh(x: Tensor) -> Tensor:
    return Tanh()(x)


class ReLU(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(np.maximum(0, x.data))

    def backward(self, *douts: Tensor):
        dout = douts[0]
        y = self._outputs()[0]
        return (y > 0) * dout

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for ReLU function."
            )


def relu(x: Tensor) -> Tensor:
    return ReLU()(x)


class Softmax(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        out_data = self._inner_forward(x.data)
        return Tensor(out_data)

    def _inner_forward(self, data: np.ndarray):
        dim_is_one = data.ndim == 1
        original_shape = data.shape

        if dim_is_one:
            data = data.reshape(1, -1)
        last_axis = len(data.shape) - 1
        max_value = np.max(data, axis=last_axis)

        # To prevent overflow of e(x), change the all input value to <0 value.
        data_normal = (data.T - max_value).T

        exp_ = np.exp(data_normal)
        exp_sum = np.sum(exp_, axis=last_axis)

        result = (exp_.T / exp_sum).T

        if dim_is_one:
            return result.reshape(original_shape)

        return result

    def backward(self, *douts: Tensor):
        # derivative equation: dx_k = dout_k * out_k - out_k * sum_over_element(dout * out)
        # reference: https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax
        dout = douts[0]
        out = self.outputs[0]()
        last_axis = len(out.shape) - 1
        dx = out * dout
        dsum = dx.sum(axis=last_axis, keepdims=True)

        return dx - out * dsum

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Softmax function."
            )


def softmax(x: Tensor) -> Tensor:
    return Softmax()(x)
