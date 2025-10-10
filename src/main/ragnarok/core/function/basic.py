import numpy as np

from src.main.ragnarok.core.function.common import (
    Function,
    FunctionVariableError,
    NotSupportedOperationException,
)
from src.main.ragnarok.core.variable import Variable


class Comparison(Function):
    def forward(self, *variables: Variable, **kwargs):
        x0, x1 = variables
        operator = kwargs["operator"]
        if operator == "eq":
            y_data = x0.data == x1.data
        elif operator == "ne":
            y_data = x0.data != x1.data
        elif operator == "lt":
            y_data = x0.data < x1.data
        elif operator == "le":
            y_data = x0.data <= x1.data
        elif operator == "gt":
            y_data = x0.data > x1.data
        elif operator == "ge":
            y_data = x0.data >= x1.data
        else:
            raise FunctionVariableError(f"Unknown operator: {operator}")
        return Variable(y_data)

    def backward(self, *douts: Variable):
        raise NotSupportedOperationException(
            "Comparison does not support backward propagation."
        )

    def _validate_variables(self, *variables: Variable, **kwargs):
        if "operator" not in kwargs:
            raise FunctionVariableError("operator is required for Comparison function.")
        operator = kwargs["operator"]
        if operator not in ["eq", "ne", "lt", "le", "gt", "ge"]:
            raise FunctionVariableError(
                "operator should be one of 'eq', 'ne', 'lt', 'le', 'gt', 'ge' for Comparison function."
            )
        var_length = len(variables)
        if var_length != 2:
            raise FunctionVariableError(
                "There should be two input variable for Comparison function."
            )


class Split(Function):
    def forward(self, *variables: Variable, num_of_splits=2, axis=0):
        x = variables[0]
        ys_data = np.split(x.data, indices_or_sections=num_of_splits, axis=axis)
        return [Variable(y_data) for y_data in ys_data]

    def backward(self, *douts: Variable):
        # TODO: Should be implemented with Function, not numpy operation
        #  to support high order differentiation
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
        # TODO: Should be implemented with Function, not numpy operation
        #  to support high order differentiation
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
