from src.main.ragnarok.core.function.common import Function
from src.main.ragnarok.core.variable import Variable


class MeanSquaredError(Function):
    def forward(self, *variables: Variable, **kwargs):
        x0, x1 = variables
        diff = x0 - x1
        length = len(diff) if diff.ndim > 0 else 1  # handle scalar
        return (diff**2).sum() / length

    def backward(self, *douts: Variable):
        dout = douts[0]

        x0, x1 = self.inputs
        diff = x0 - x1

        length = len(diff) if diff.ndim > 0 else 1
        dx0 = 2 * diff * dout / length
        dx1 = -dx0

        return dx0, dx1

    def _validate_variables(self, *variables: Variable, **kwargs):
        if len(variables) != 2:
            raise ValueError("MeanSquaredError requires 2 variables")
        if variables[0].shape != variables[1].shape:
            raise ValueError("MeanSquaredError requires the same shape variables")
