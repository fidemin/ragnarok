import numpy as np

from ragnarok.core.function.common import (
    Function,
    FunctionVariableError,
    NotSupportedOperationException,
)
from ragnarok.core.tensor import Tensor


class Comparison(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        x0, x1 = tensors
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
        return Tensor(y_data)

    def backward(self, *douts: Tensor):
        raise NotSupportedOperationException(
            "Comparison does not support backward propagation."
        )

    def _validate_variables(self, *variables: Tensor, **kwargs):
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
                "There should be two input tensor for Comparison function."
            )


class Split(Function):
    def forward(self, *variables: Tensor, num_of_splits=2, axis=0):
        x = variables[0]
        ys_data = np.split(x.data, indices_or_sections=num_of_splits, axis=axis)
        return [Tensor(y_data) for y_data in ys_data]

    def backward(self, *douts: Tensor):
        # TODO: Should be implemented with Function, not numpy operation
        #  to support high order differentiation
        douts_data = [dout.data for dout in douts]
        dx = np.concatenate(douts_data, axis=self.kwargs["axis"])
        return Tensor(dx)

    def _validate_variables(self, *variables: Tensor, **kwargs):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Split function."
            )


class Reshape(Function):
    def forward(self, *variables: Tensor, **kwargs):
        shape = kwargs["shape"]
        x = variables[0]
        y = x.data.reshape(shape)
        return Tensor(y)

    def backward(self, *douts: Tensor):
        # TODO: Should be implemented with Function, not numpy operation
        #  to support high order differentiation
        dx = douts[0].data.reshape(self.inputs[0].shape)
        return Tensor(dx)

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if "shape" not in kwargs:
            raise FunctionVariableError("shape is required for Reshape function.")
        shape = kwargs["shape"]
        if not isinstance(shape, tuple):
            raise FunctionVariableError("shape should be a tuple for Reshape function.")
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Reshape function."
            )


class Transpose(Function):
    def forward(self, *variables: Tensor, **kwargs):
        transpose = kwargs.get("transpose", None)
        x = variables[0]
        if transpose is None:
            y_var = x.data.T
        else:
            y_var = x.data.transpose(transpose)
        return Tensor(y_var)

    def backward(self, *douts: Tensor):
        dout = douts[0]
        transpose = self.kwargs.get("transpose", None)
        dx = Transpose()(dout, transpose=transpose)
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
        transpose = kwargs.get("transpose", None)
        if transpose is not None and not isinstance(transpose, tuple):
            raise FunctionVariableError(
                "transpose should be a tuple for Transpose function."
            )

        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Transpose function."
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
    def forward(self, *variables: Tensor, **kwargs):
        shape = kwargs["shape"]
        x = variables[0]
        try:
            y_var = np.broadcast_to(x.data, shape)
        except ValueError as e:
            raise FunctionVariableError(
                f"Can not broadcast {x.shape} to {shape}: {str(e)}"
            )
        return Tensor(y_var)

    def backward(self, *douts: Tensor):
        to_shape = self.inputs[0].shape
        dx = sum_to(douts[0], to_shape)
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
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
                "There should be one input tensor for BroadcastTo function."
            )


class SumTo(Function):
    def forward(self, *variables: Tensor, **kwargs):
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

        return Tensor(y_var)

    def backward(self, *douts: Tensor):
        to_shape = self.inputs[0].shape
        dx = BroadcastTo()(douts[0], shape=to_shape)
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if "shape" not in kwargs:
            raise FunctionVariableError("shape is required for SumTo function.")
        shape = kwargs["shape"]
        if not isinstance(shape, tuple):
            raise FunctionVariableError("shape should be a tuple for SumTo function.")
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for SumTo function."
            )


def sum_to(x: Tensor, shape: tuple) -> Tensor:
    return SumTo()(x, shape=shape)


class Sum(Function):
    def forward(self, *variables: Tensor, **kwargs):
        axis = kwargs.get("axis", None)
        keepdims = kwargs.get("keepdims", False)
        x = variables[0]
        y = np.sum(x.data, axis=axis, keepdims=keepdims)
        return Tensor(y)

    def backward(self, *douts: Tensor):
        dout = douts[0]

        input_ = self.inputs[0]
        shape = input_.shape

        dx = BroadcastTo()(dout, shape=shape)
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
        var_length = len(variables)
        if var_length != 1:
            raise FunctionVariableError(
                "There should be one input tensor for Sum function."
            )


class GetItem(Function):
    def forward(self, *variables: Tensor, **kwargs):
        x = variables[0]
        index = kwargs["index"]
        y = x.data[index]
        return Tensor(y)

    def backward(self, *douts: Tensor):
        dout = douts[0]
        dx = GetItemGrad()(
            dout, to_shape=self.inputs[0].shape, index=self.kwargs["index"]
        )
        return dx

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if len(variables) != 1:
            raise FunctionVariableError(
                "There should be one input tensor for GetItem function."
            )

        if "index" not in kwargs:
            raise FunctionVariableError("`index` is required for GetItem function.")


def get_item(x: Tensor, index) -> Tensor:
    return GetItem()(x, index=index)


class GetItemGrad(Function):
    def forward(self, *variables: Tensor, **kwargs):
        x = variables[0]
        to_shape = kwargs["to_shape"]
        index = kwargs["index"]

        y = np.zeros(to_shape, dtype=x.dtype)
        np.add.at(y, index, x.data)
        return Tensor(y)

    def backward(self, *douts: Tensor):
        dout = douts[0]
        return get_item(dout, index=self.kwargs["index"])

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if len(variables) != 1:
            raise FunctionVariableError(
                "There should be one input tensor for GetItemGrad function."
            )

        if "to_shape" not in kwargs:
            raise FunctionVariableError(
                "`to_shape` is required for GetItemGrad function."
            )

        if "index" not in kwargs:
            raise FunctionVariableError("`index` is required for GetItemGrad function.")
