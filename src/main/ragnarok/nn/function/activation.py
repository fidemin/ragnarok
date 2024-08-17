import numpy as np

from src.main.ragnarok.core.function import exp
from src.main.ragnarok.core.function.common import Function, FunctionVariableError
from src.main.ragnarok.core.variable import Variable


class Sigmoid(Function):
    def forward(self, *variables: Variable, **kwargs):
        x = variables[0]
        return 1 / (1 + exp(-x))

    def backward(self, *douts: Variable):
        dout = douts[0]
        y = self._outputs()[0]
        return dout * y * (1.0 - y)

    def _validate_variables(self, *variables: Variable, **kwargs):
        if len(variables) != 1:
            raise ValueError("Sigmoid requires 1 variable")


def sigmoid(x: Variable) -> Variable:
    return Sigmoid()(x)


class Tanh(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.tanh(x.data))

    def backward(self, *douts: Variable):
        dout = douts[0]
        y = self._outputs()[0]
        return (1 - y**2) * dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Tanh function."
            )


def tanh(x: Variable) -> Variable:
    return Tanh()(x)


class ReLU(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.maximum(0, x.data))

    def backward(self, *douts: Variable):
        dout = douts[0]
        y = self._outputs()[0]
        return (y > 0) * dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for ReLU function."
            )


def relu(x: Variable) -> Variable:
    return ReLU()(x)


class Softmax(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        out_data = self._inner_forward(x.data)
        return Variable(out_data)

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

    def backward(self, *douts: Variable):
        # derivative equation: dx_k = dout_k * out_k - out_k * sum_over_element(dout * out)
        # reference: https://www.mldawn.com/back-propagation-with-cross-entropy-and-softmax
        dout = douts[0]
        out = self.outputs[0]()
        last_axis = len(out.shape) - 1
        dx = out * dout
        dsum = dx.sum(axis=last_axis, keepdims=True)

        return dx - out * dsum

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Softmax function."
            )
