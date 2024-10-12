import numpy as np

from src.main.ragnarok.core.function import Function, FunctionVariableError, sum_to
from src.main.ragnarok.core.variable import Variable


class Square(Function):
    """
    Square function returns square of values in Variable.
    """

    def forward(self, *variables: Variable):
        x_var = variables[0]
        out = np.square(x_var.data)
        out_var = Variable(out)
        return out_var

    def backward(self, *douts: Variable):
        dout = douts[0]
        x_var = self.inputs[0]
        dx = 2 * x_var
        grad = dx * dout
        return grad

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError(
                "There should be one input variable for Square function."
            )


class Exp(Function):
    """
    Exp function returns exponential of Variable.
    """

    def forward(self, *variables: Variable):
        x_var = variables[0]
        out = np.exp(x_var.data)
        out_var = Variable(out)
        return out_var

    def backward(self, *douts: Variable):
        dout = douts[0]
        out = self._outputs()[0]
        grad = out * dout
        return grad

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError(
                "There should be one input variable for Exp function."
            )


def exp(x: Variable) -> Variable:
    return Exp()(x)


class Add(Function):
    def forward(self, *variables: Variable):
        x0, x1 = variables
        y = x0.data + x1.data  # NOTE: If shape is different, numpy will broadcast
        return Variable(y)

    def backward(self, *dout: Variable):
        dout = dout[0]

        # To handle broadcast, sum to the shape of input variable
        dx0 = sum_to(dout, self.inputs[0].shape)
        dx1 = sum_to(dout, self.inputs[1].shape)
        return dx0, dx1

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for Add function."
            )


class Subtract(Function):
    def forward(self, *variables: Variable):
        x0, x1 = variables
        y = x0.data - x1.data
        return Variable(y)

    def backward(self, dout: Variable):
        return dout, -dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for Subtract function."
            )


class Multiply(Function):
    def forward(self, *variables: Variable, **kwargs):
        x0, x1 = variables
        y = x0.data * x1.data
        return Variable(y)

    def backward(self, dout: Variable):
        dx0 = self.inputs[1] * dout
        dx1 = self.inputs[0] * dout
        return dx0, dx1

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for Multiply function."
            )


class Divide(Function):
    def forward(self, *variables: Variable):
        x0, x1 = variables
        y = x0.data / x1.data
        return Variable(y)

    def backward(self, dout: Variable):
        dx0 = dout / self.inputs[1]
        dx1 = -dout * self.inputs[0] / (self.inputs[1] ** 2)
        return dx0, dx1

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for Divide function."
            )


class Pow(Function):
    def forward(self, *variables: Variable, **kwargs):
        x = variables[0]
        power = kwargs["power"]
        y = x.data**power
        return Variable(y)

    def backward(self, dout: Variable):
        x = self.inputs[0]
        power = self.kwargs["power"]
        dx = power * (x ** (power - 1)) * dout
        return dx

    def _validate_variables(self, *variables: Variable, **kwargs):
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
                "There should be two input variable for Pow function."
            )


class Negative(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(-x.data)

    def backward(self, *douts: Variable):
        return -douts[0]

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Negative function."
            )


class Sin(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.sin(x.data))

    def backward(self, dout: Variable):
        x = self.inputs[0]
        return cos(x) * dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Sin function."
            )


def sin(x: Variable) -> Variable:
    return Sin()(x)


class Cos(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.cos(x.data))

    def backward(self, dout: Variable):
        x = self.inputs[0]
        return -sin(x) * dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Cos function."
            )


def cos(x: Variable) -> Variable:
    return Cos()(x)


class Log(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.log(x.data))

    def backward(self, dout: Variable):
        x = self.inputs[0]
        return dout / x

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Log function."
            )


def log(x: Variable) -> Variable:
    return Log()(x)


class MatMul(Function):
    def forward(self, *variables: Variable):
        x0, x1 = variables
        y = np.dot(x0.data, x1.data)
        return Variable(y)

    def backward(self, *douts: Variable):
        x0, x1 = self.inputs
        dout = douts[0]
        dx0 = matmul(dout, x1.T)
        dx1 = matmul(x0.T, dout)
        return dx0, dx1

    def _validate_variables(self, *variables: Variable, **kwargs):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for MatMul function."
            )


def matmul(x0: Variable, x1: Variable) -> Variable:
    return MatMul()(x0, x1)
