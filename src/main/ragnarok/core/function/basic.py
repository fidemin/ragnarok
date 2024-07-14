import numpy as np

from src.main.ragnarok.core.function.common import Function
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


class Split(Function):
    def forward(self, *variables: Variable, num_of_splits=2, axis=0):
        x = variables[0]
        ys_data = np.split(x.data, indices_or_sections=num_of_splits, axis=axis)
        return [Variable(y_data) for y_data in ys_data]

    def backward(self, *douts: Variable):
        # TODO: Should be implemented with Function not numpy operation
        douts_data = [dout.data for dout in douts]
        dx = np.concatenate(douts_data, axis=self.kwargs["axis"])
        return Variable(dx)

    def _validate_variables(self, *variables: Variable, **kwargs):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Split function."
            )


class Reshape(Function):
    def forward(self, *variables: Variable, **kwargs):
        shape = kwargs["shape"]
        x = variables[0]
        y = x.data.reshape(shape)
        return Variable(y)

    def backward(self, *douts: Variable):
        # TODO: Should be implemented with Function not numpy operation
        dx = douts[0].data.reshape(self.inputs[0].shape)
        return Variable(dx)

    def _validate_variables(self, *variables: Variable, **kwargs):
        if "shape" not in kwargs:
            raise FunctionVariableError("shape is required for Reshape function.")
        shape = kwargs["shape"]
        if not isinstance(shape, tuple):
            raise FunctionVariableError("shape should be a tuple for Reshape function.")
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Reshape function."
            )


class Transpose(Function):
    def forward(self, *variables: Variable, **kwargs):
        transpose = kwargs.get("transpose", None)
        x = variables[0]
        if transpose is None:
            y_var = x.data.T
        else:
            y_var = x.data.transpose(transpose)
        return Variable(y_var)

    def backward(self, *douts: Variable):
        dout = douts[0]
        transpose = self.kwargs.get("transpose", None)
        dx = Transpose()(dout, transpose=transpose)
        return dx

    def _validate_variables(self, *variables: Variable, **kwargs):
        transpose = kwargs.get("transpose", None)
        if transpose is not None and not isinstance(transpose, tuple):
            raise FunctionVariableError(
                "transpose should be a tuple for Transpose function."
            )

        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Transpose function."
            )


def _find_axis_to_for_sum_to(from_shape: tuple, to_shape: tuple) -> (tuple, tuple):
    """
    Find axis to sum from shape to to_shape.
    Can deal with following cases.
    1. from_shape is larger than to_shape and the last elements of from_shape is equal to to_shape.
    e.g. (2, 3, 4, 5) -> (3, 4, 5)
    2. from_shape is equal to to_shape and the summed shape elements are 1 in to_shape.
    e.g. (2, 3, 4, 5) -> (2, 1, 4, 5)
    3. Combination of 1 and 2.
    e.g. (2, 3, 4, 5) -> (1, 4, 5)
    e.g. (2, 3, 4, 5) -> (3, 1, 5)

    Args:
        from_shape: shape to sum
        to_shape: shape to sum to

    Returns:
        axis_without_keepdims: axis to sum without keepdims
        axis_with_keepdims: axis to sum with keepdims
    """

    from_shape_len = len(from_shape)
    to_shape_len = len(to_shape)
    diff_len = from_shape_len - to_shape_len
    if diff_len < 0:
        raise FunctionVariableError(
            f"The length of {from_shape} should be smaller than or equal to {to_shape}."
        )

    axis_without_keepdims = tuple()
    if diff_len:
        axis_without_keepdims = tuple(range(0, diff_len))

    axis_with_keepdims = []
    for i, (x1, x2) in enumerate(zip(from_shape[diff_len:], to_shape)):
        if x1 != x2 and x2 != 1:
            raise FunctionVariableError(
                f"The shape {from_shape} can not be summed to {to_shape}."
            )

        if x2 == 1:
            axis_with_keepdims.append(i)

    return axis_without_keepdims, tuple(axis_with_keepdims)


class BroadcastTo(Function):
    def forward(self, *variables: Variable, **kwargs):
        shape = kwargs["shape"]
        x = variables[0]
        try:
            y_var = np.broadcast_to(x.data, shape)
        except ValueError as e:
            raise FunctionVariableError(
                f"Can not broadcast {x.shape} to {shape}: {str(e)}"
            )
        return Variable(y_var)

    def backward(self, *douts: Variable):
        to_shape = self.inputs[0].shape
        dx = sum_to(douts[0], to_shape)
        return dx

    def _validate_variables(self, *variables: Variable, **kwargs):
        if "shape" not in kwargs:
            raise FunctionVariableError("shape is required for BroadcastTo function.")
        shape = kwargs["shape"]
        if not isinstance(shape, tuple):
            raise FunctionVariableError(
                "shape should be a tuple for BroadcastTo function."
            )
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for BroadcastTo function."
            )


class SumTo(Function):
    def forward(self, *variables: Variable, **kwargs):
        shape = kwargs["shape"]
        x = variables[0]
        axis_without_keepdims, axis_with_keepdims = _find_axis_to_for_sum_to(
            x.shape, shape
        )

        y_var = x.data

        if axis_without_keepdims:
            y_var = np.sum(y_var, axis=axis_without_keepdims, keepdims=False)

        if axis_with_keepdims:
            y_var = np.sum(y_var, axis=axis_with_keepdims, keepdims=True)

        return Variable(y_var)

    def backward(self, *douts: Variable):
        to_shape = self.inputs[0].shape
        dx = BroadcastTo()(douts[0], shape=to_shape)
        return dx

    def _validate_variables(self, *variables: Variable, **kwargs):
        if "shape" not in kwargs:
            raise FunctionVariableError("shape is required for SumTo function.")
        shape = kwargs["shape"]
        if not isinstance(shape, tuple):
            raise FunctionVariableError("shape should be a tuple for SumTo function.")
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for SumTo function."
            )


def sum_to(x: Variable, shape: tuple) -> Variable:
    return SumTo()(x, shape=shape)


class Sum(Function):
    def forward(self, *variables: Variable, **kwargs):
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        x = variables[0]
        y = np.sum(x.data, axis=axis, keepdims=keepdims)
        return Variable(y)

    def backward(self, *douts: Variable):
        dout = douts[0]

        input_ = self.inputs[0]
        shape = input_.shape

        dx = BroadcastTo()(dout, shape=shape)
        return dx

    def _validate_variables(self, *variables: Variable, **kwargs):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input variable for Sum function."
            )


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


class FunctionVariableError(RuntimeError):
    pass
