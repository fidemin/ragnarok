import numpy as np

from ragnarok.core.variable import Variable


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def __call__(self, *inputs: Variable):
        self._validate_variables(*inputs)
        outputs = self.forward(*inputs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

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

    def backward(self, *douts: Variable):
        dout = douts[0]
        x_var = self.inputs[0]
        dx = 2 * x_var.data
        grad = Variable(dx * dout.data)
        return grad,

    def forward(self, *variables: Variable):
        x_var = variables[0]
        out = np.square(x_var.data)
        out_var = Variable(out)
        return out_var

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Square function.')


class Exp(Function):
    """
    Exp function returns exponential of Variable.
    """

    def backward(self, *douts: Variable):
        dout = douts[0]
        out = self.outputs[0]
        grad = Variable(out.data * dout.data)
        return grad,

    def forward(self, *variables: Variable):
        x_var = variables[0]
        out = np.exp(x_var.data)
        out_var = Variable(out)
        return out_var

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError('There should be one input variable for Exp function.')


class Add(Function):
    def forward(self, *variables: Variable):
        x0, x1 = variables
        y = x0.data + x1.data
        return Variable(y)

    def backward(self, dout: Variable):
        return dout, dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError('There should be two input variable for Add function.')


class FunctionVariableError(RuntimeError):
    pass
