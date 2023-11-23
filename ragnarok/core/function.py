import numpy as np

from ragnarok.core.variable import Variable


class Function:
    def __init__(self):
        self.inputs = None
        self.output = None

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
    Square function returns square of values in Variable.
    """

    def backward(self, dout: Variable):
        x_var = self.inputs[0]
        dx = 2 * x_var.data
        grad = Variable(dx * dout.data)
        return grad,

    def forward(self, *variables: Variable):
        x_var = variables[0]
        output_ = np.square(x_var.data)
        out_var = Variable(output_, creator=self)
        self.inputs: tuple[Variable] = variables
        self.output: Variable = out_var
        return out_var

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Square function.')


class Exp(Function):
    """
    Exp function returns exponential of Variable.
    """

    def backward(self, dout: Variable):
        out = self.output
        grad = Variable(out.data * dout.data)
        return (grad,)

    def forward(self, *variables: Variable):
        x_var = variables[0]
        out = np.exp(x_var.data)
        out_var = Variable(out, creator=self)
        self.inputs = variables
        self.output = out_var
        return out_var

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Exp function.')


class FunctionVariableError(RuntimeError):
    pass
