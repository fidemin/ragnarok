import heapq
from typing import Optional

import numpy as np

from ragnarok.core.config import using_backprop
from ragnarok.core.utils.id_generator import IncrementalIdGenerator


class Tensor:
    __array_priority__ = 200

    def __init__(self, data: int | float | list | np.ndarray | np.generic, name=None):
        """
        Args:
            data: numpy array or int or float
        Attributes:
            data: original data
            creator: Function instance which generates this tensor
            grad: gradient of this tensor
            gen: generation of this tensor. used for backpropagation
        """

        if not (isinstance(data, (int, float, list, np.ndarray, np.generic))):
            raise VariableError(
                f"data should be numpy array or int or float. given: {type(data)}"
            )

        if isinstance(data, (int, float)):
            data = np.array(data)

        if isinstance(data, list):
            data = np.array(data)

        self._name = name
        self._data = data
        self._creator = None
        self._grad: Optional[Tensor] = None
        self._gen = 0

    def __len__(self):
        return self._data.shape[0] if self._data.shape else 0

    def __repr__(self):
        return f"Tensor({str(self._data)})"

    def __mul__(self, other):
        from ragnarok.core.function.math import Multiply

        return Multiply()(self, other)

    def __rmul__(self, other):
        from ragnarok.core.function.math import Multiply

        return Multiply()(other, self)

    def __add__(self, other):
        from ragnarok.core.function.math import Add

        return Add()(self, other)

    def __iadd__(self, other):
        from ragnarok.core.function.math import InplaceAdd

        return InplaceAdd()(self, other)

    def __radd__(self, other):
        from ragnarok.core.function.math import Add

        return Add()(other, self)

    def __sub__(self, other):
        from ragnarok.core.function.math import Subtract

        return Subtract()(self, other)

    def __rsub__(self, other):
        from ragnarok.core.function.math import Subtract

        return Subtract()(other, self)

    def __truediv__(self, other):
        from ragnarok.core.function.math import Divide

        return Divide()(self, other)

    def __rtruediv__(self, other):
        from ragnarok.core.function.math import Divide

        return Divide()(other, self)

    def __neg__(self):
        from ragnarok.core.function.math import Negative

        return Negative()(self)

    def __pow__(self, power, modulo=None):
        from ragnarok.core.function.math import Pow

        return Pow()(self, power=power)

    def __eq__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="eq")

    def __ne__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="ne")

    def __lt__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="lt")

    def __le__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="le")

    def __gt__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="gt")

    def __ge__(self, other) -> "Tensor":
        from ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="ge")

    def __getitem__(self, index):
        from ragnarok.core.function.matrix import get_item

        return get_item(self, index)

    def set_creator(self, creator):
        self._creator = creator
        self._gen = creator.gen + 1

    @property
    def data(self) -> int | float | np.ndarray | np.generic:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def creator(self):
        return self._creator

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def gen(self):
        return self._gen

    @gen.setter
    def gen(self, value):
        self._gen = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype.name

    def astype(self, dtype) -> "Tensor":
        return Tensor(self._data.astype(dtype))

    def copy(self):
        return Tensor(self._data.copy())

    def clear_grad(self):
        self._grad = None

    def release(self):
        self._data = None

    def reshape(self, *shape):
        from ragnarok.core.function import Reshape

        # if shape is tuple of tuple, convert to tuple
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return Reshape()(self, shape=shape)

    def transpose(self, *transpose: Optional[int | tuple]):
        from ragnarok.core.function import Transpose

        if not transpose:
            return Transpose()(self)

        if len(transpose) == 1 and isinstance(transpose[0], tuple):
            transpose = transpose[0]

        return Transpose()(self, transpose=transpose)

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        from ragnarok.core.function import Sum

        return Sum()(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        divisor = self.data.size if axis is None else self.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) / divisor

    @property
    def T(self):
        return self.transpose()

    def argmax(self, *, axis=None):
        # TODO: implement argmax function
        return Tensor(np.argmax(self.data, axis=axis))

    def backward(self, retain_grad=False, enable_double_backprop=False):
        """
        Backward propagation to calculate gradients of all tensors which are used to create this tensor.

        Args:
            retain_grad: Whether to retain the gradient of intermediate tensors during backpropagation.
            enable_double_backprop: Whether to enable double backpropagation for higher order differentiation.

        Returns:
            None
        """
        if self._creator is None:
            raise VariableError(
                "The creator of this tensor is None. backward propagation is not supported."
            )

        if self._grad is None:
            # initialize with first grad as 1 (i.e. dL/dL)
            self._grad = Tensor(np.ones_like(self.data))

        # higher generation popped first
        function_queue = []
        # idx used to prevent crash in heapq operation: if gen is same, use idx to compare (without idx, it raises error)
        # gen (generation) is always non-negative, so -gen is used to pop higher gen first
        id_generator = IncrementalIdGenerator()
        heapq.heappush(
            function_queue, ((-self.creator.gen, id_generator.next()), self.creator)
        )
        visited = {self.creator}

        # DFS to iterate all related tensors: use pop and append
        while function_queue:
            _, function = heapq.heappop(function_queue)

            doutputs = [output().grad for output in function.outputs]

            # (1) For first-order differentiation, backpropagation of grad tensor is not needed
            # -> enable_double_backprop=False is recommended
            # (2) For higher-order differentiation, backpropagation of grad tensor is needed
            # -> enable_double_backprop=True is needed.
            # This config is used in Function.__call__ to reserve computational graph for grad tensors
            with using_backprop(enable_double_backprop):
                dinputs = function.backward(*doutputs)
                if not isinstance(dinputs, tuple):
                    dinputs = (dinputs,)

                inputs = function.inputs

                for input_, dinput in zip(inputs, dinputs):
                    if input_.grad is not None:
                        # For the function has more than one input and same inputs are used for the function
                        # e.g. Add()(x, x)
                        # NOTE: do not use `input_.grad += dinput`, because it is inplace operation and may cause error in some cases
                        input_.grad = input_.grad + dinput
                    else:
                        input_.grad = dinput

                    if input_.creator is not None:
                        next_function = input_.creator
                        if next_function in visited:
                            continue
                        heapq.heappush(
                            function_queue,
                            ((-next_function.gen, id_generator.next()), next_function),
                        )
                        visited.add(next_function)

            if not retain_grad:
                # In general case, output grad is intermittent tensor and it can be released to save memory
                for output in function.outputs:
                    output().grad = None


class VariableError(RuntimeError):
    pass


def to_variable(x: int | float | np.ndarray | np.generic | Tensor) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def zeros_like(x: Tensor) -> Tensor:
    return Tensor(np.zeros_like(x.data, dtype=x.dtype))


def ones_like(x: Tensor) -> Tensor:
    return Tensor(np.ones_like(x.data, dtype=x.dtype))


def zeros(shape, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype))
