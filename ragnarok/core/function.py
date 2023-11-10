import numpy as np

from ragnarok.core.variable import Variable


class Function:
    def __init__(self):
        self._cache = {}

    def __call__(self, *variables: Variable):
        self._validate_variables(*variables)
        return self.forward(*variables)

    def backward(self, dout: Variable):
        raise NotImplementedError("Function.backward is not implemented")

    def forward(self, *variables: Variable):
        raise NotImplementedError("Function._forward is not implemented")

    def _validate_variables(self, *variables: Variable):
        raise NotImplementedError("Function._validate_input is not implemented")


class Square(Function):
    """
    Sqauare function returns square of values in Variable.
    """

    def backward(self, dout: Variable):
        x_var = self._cache['x_var']
        dx = 2 * x_var.data
        return Variable(dx * dout.data)

    def forward(self, *variables: Variable):
        x_var = variables[0]
        output_ = np.square(x_var.data)
        self._cache['x_var'] = x_var
        return Variable(output_)

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Square function.')


class FunctionVariableError(RuntimeError):
    pass