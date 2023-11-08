import numpy as np

from ragnarok.core.variable import Variable


class Function:
    def __call__(self, *variables: Variable):
        self._validate_variables(*variables)
        return self._forward(*variables)

    def _forward(self, *variables: Variable):
        raise NotImplementedError("Function._forward is not implemented")

    def _validate_variables(self, *variables: Variable):
        raise NotImplementedError("Function._validate_input is not implemented")


class Square(Function):
    """
    Sqauare function returns square of values in Variable.
    """

    def _forward(self, *variables: Variable):
        var = variables[0]
        output_ = np.square(var.data)
        return Variable(output_)

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Square function.')


class FunctionVariableError(RuntimeError):
    pass
