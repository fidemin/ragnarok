from src.main.ragnarok.core.function import exp
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
