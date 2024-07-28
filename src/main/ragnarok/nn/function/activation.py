import numpy as np

from src.main.ragnarok.core.function import exp, FunctionVariableError
from src.main.ragnarok.core.function.common import Function
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
