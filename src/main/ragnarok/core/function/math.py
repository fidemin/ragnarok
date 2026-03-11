import numpy as np

from src.main.ragnarok.core.function import (
    Function,
    FunctionVariableError,
    sum_to,
    NotSupportedOperationException,
)
from src.main.ragnarok.core.tensor import Tensor


class Square(Function):
    """
    Square function returns square of values in Tensor.
    """

    def forward(self, *variables: Tensor):
        x_var = variables[0]
        out = np.square(x_var.data)
        out_var = Tensor(out)
        return out_var

    def backward(self, *douts: Tensor):
        dout = douts[0]
        x_var = self.inputs[0]
        dx = 2 * x_var
        grad = dx * dout
        return grad

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError(
                "There should be one input tensor for Square function."
            )


class Exp(Function):
    """
    Exp function returns exponential of Tensor.
    """

    def forward(self, *variables: Tensor):
        x_var = variables[0]
        out = np.exp(x_var.data)
        out_var = Tensor(out)
        return out_var

    def backward(self, *douts: Tensor):
        dout = douts[0]
        out = self._outputs()[0]
        grad = out * dout
        return grad

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError(
                "There should be one input tensor for Exp function."
            )


def exp(x: Tensor) -> Tensor:
    return Exp()(x)


class Add(Function):
    def forward(self, *variables: Tensor):
        x0, x1 = variables
        y = x0.data + x1.data  # NOTE: If shape is different, numpy will broadcast
        return Tensor(y)

    def backward(self, *dout: Tensor):
        dout = dout[0]

        # To handle broadcast, sum to the shape of input tensor
        dx0 = sum_to(dout, self.inputs[0].shape)
        dx1 = sum_to(dout, self.inputs[1].shape)
        return dx0, dx1

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for Add function."
            )


class InplaceAdd(Function):
    def forward(self, *variables: Tensor):
        x0, x1 = variables
        x0.data += x1.data  # numpy inplace add
        return x0

    def backward(self, *dout: Tensor):
        # TODO: implement inplace add backward.
        #   It can be issue only when x0 is used in grad calculation in other tensor's backward propagation.
        raise NotSupportedOperationException(
            "InplaceAdd does not support backward propagation."
        )

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for InplaceAdd function."
            )


class Subtract(Function):
    def forward(self, *variables: Tensor):
        x0, x1 = variables
        y = x0.data - x1.data
        return Tensor(y)

    def backward(self, dout: Tensor):
        return dout, -dout

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for Subtract function."
            )


class Multiply(Function):
    def forward(self, *variables: Tensor, **kwargs):
        x0, x1 = variables
        y = x0.data * x1.data
        return Tensor(y)

    def backward(self, dout: Tensor):
        dx0 = self.inputs[1] * dout
        dx1 = self.inputs[0] * dout
        return dx0, dx1

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for Multiply function."
            )


class Divide(Function):
    def forward(self, *variables: Tensor):
        x0, x1 = variables
        y = x0.data / x1.data
        return Tensor(y)

    def backward(self, dout: Tensor):
        dx0 = dout / self.inputs[1]
        dx1 = -dout * self.inputs[0] / (self.inputs[1] ** 2)
        return dx0, dx1

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for Divide function."
            )


class Pow(Function):
    def forward(self, *variables: Tensor, **kwargs):
        x = variables[0]
        power = kwargs["power"]
        y = x.data**power
        return Tensor(y)

    def backward(self, dout: Tensor):
        x = self.inputs[0]
        power = self.kwargs["power"]
        dx = power * (x ** (power - 1)) * dout
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if "power" not in kwargs:
            raise FunctionVariableError("power is required for Pow function.")
        power = kwargs["power"]
        if not isinstance(power, (int, float)):
            raise FunctionVariableError(
                "power should be int or float for Pow function."
            )
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be two input tensor for Pow function."
            )


class Negative(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(-x.data)

    def backward(self, *douts: Tensor):
        return -douts[0]

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Negative function."
            )


class Sin(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(np.sin(x.data))

    def backward(self, dout: Tensor):
        x = self.inputs[0]
        return cos(x) * dout

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Sin function."
            )


def sin(x: Tensor) -> Tensor:
    return Sin()(x)


class Cos(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(np.cos(x.data))

    def backward(self, dout: Tensor):
        x = self.inputs[0]
        return -sin(x) * dout

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Cos function."
            )


def cos(x: Tensor) -> Tensor:
    return Cos()(x)


class Log(Function):
    def forward(self, *variables: Tensor):
        x = variables[0]
        return Tensor(np.log(x.data))

    def backward(self, dout: Tensor):
        x = self.inputs[0]
        return dout / x

    def _validate_variables(self, *variables: Tensor):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Log function."
            )


def log(x: Tensor) -> Tensor:
    return Log()(x)


class MatMul(Function):
    def forward(self, *variables: Tensor):
        x0, x1 = variables
        y = np.dot(x0.data, x1.data)
        return Tensor(y)

    def backward(self, *douts: Tensor):
        x0, x1 = self.inputs
        dout = douts[0]
        dx0 = matmul(dout, x1.T)
        dx1 = matmul(x0.T, dout)
        return dx0, dx1

    def _validate_variables(self, *variables: Tensor, **kwargs):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input tensor for MatMul function."
            )


def matmul(x0: Tensor, x1: Tensor) -> Tensor:
    return MatMul()(x0, x1)
