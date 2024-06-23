import weakref

import numpy as np

from src.main.ragnarok.core.config import Config
from src.main.ragnarok.core.variable import Variable, to_variable


class Function:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.gen = None
        self._cache = {}
        self.kwargs = {}

    def __call__(
        self, *inputs: int | float | np.ndarray | np.generic | Variable, **kwargs
    ):
        inputs = [to_variable(input_) for input_ in inputs]
        self._validate_variables(*inputs)
        outputs = self.forward(*inputs, **kwargs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        if Config.enable_backprop:
            # these are only used for backpropagation
            this_gen = max([input_.gen for input_ in inputs])
            self.gen = this_gen

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            self.kwargs = kwargs

        return outputs if len(outputs) > 1 else outputs[0]

    def backward(self, *douts: Variable):
        raise NotImplementedError("Function.backward is not implemented")

    def forward(self, *variables: Variable, **kwargs):
        raise NotImplementedError("Function._forward is not implemented")

    def _validate_variables(self, *variables: Variable):
        raise NotImplementedError("Function._validate_input is not implemented")


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
        out = self.outputs[0]()
        grad = out * dout
        return grad

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length == 0 or var_length > 1:
            raise FunctionVariableError(
                "There should be one input variable for Exp function."
            )


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

    def _validate_variables(self, *variables: Variable):
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


class Tanh(Function):
    def forward(self, *variables: Variable):
        x = variables[0]
        return Variable(np.tanh(x.data))

    def backward(self, *douts: Variable):
        dout = douts[0]
        y = self.outputs[0]()  # weak reference
        return (1 - y**2) * dout

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Tanh function."
            )


class Split(Function):
    def forward(self, *variables: Variable, num_of_splits=2, axis=0):
        x = variables[0]
        ys_data = np.split(x.data, indices_or_sections=num_of_splits, axis=axis)
        return [Variable(y_data) for y_data in ys_data]

    def backward(self, *douts: Variable):
        douts_data = [dout.data for dout in douts]
        dx = np.concatenate(douts_data, axis=self.kwargs["axis"])
        return Variable(dx)

    def _validate_variables(self, *variables: Variable):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Split function."
            )


class FunctionVariableError(RuntimeError):
    pass
