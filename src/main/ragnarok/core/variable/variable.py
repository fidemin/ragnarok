import heapq
from typing import Optional

import numpy as np

from src.main.ragnarok.core.config import using_config


class Variable:
    __array_priority__ = 200

    def __init__(self, data: int | float | list | np.ndarray | np.generic, name=None):
        """
        Args:
            data: numpy array or int or float
        Attributes:
            data: original data
            creator: Function instance which generate this variable
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
        self._grad: Optional[Variable] = None
        self._gen = 0

    def __len__(self):
        return self._data.shape[0] if self._data.shape else 0

    def __repr__(self):
        return f"Variable({str(self._data)})"

    def __mul__(self, other):
        from src.main.ragnarok.core.function import Multiply

        return Multiply()(self, other)

    def __rmul__(self, other):
        from src.main.ragnarok.core.function import Multiply

        return Multiply()(other, self)

    def __add__(self, other):
        from src.main.ragnarok.core.function import Add

        return Add()(self, other)

    def __radd__(self, other):
        from src.main.ragnarok.core.function import Add

        return Add()(other, self)

    def __sub__(self, other):
        from src.main.ragnarok.core.function import Subtract

        return Subtract()(self, other)

    def __rsub__(self, other):
        from src.main.ragnarok.core.function import Subtract

        return Subtract()(other, self)

    def __truediv__(self, other):
        from src.main.ragnarok.core.function import Divide

        return Divide()(self, other)

    def __rtruediv__(self, other):
        from src.main.ragnarok.core.function import Divide

        return Divide()(other, self)

    def __neg__(self):
        from src.main.ragnarok.core.function import Negative

        return Negative()(self)

    def __pow__(self, power, modulo=None):
        from src.main.ragnarok.core.function import Pow

        return Pow()(self, power=power)

    def __eq__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="eq")

    def __ne__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="ne")

    def __lt__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="lt")

    def __le__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="le")

    def __gt__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="gt")

    def __ge__(self, other) -> "Variable":
        from src.main.ragnarok.core.function import Comparison

        return Comparison()(self, other, operator="ge")

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

    def copy(self):
        return Variable(self._data.copy())

    def clear_grad(self):
        self._grad = None

    def release(self):
        self._data = None

    def reshape(self, *shape):
        from src.main.ragnarok.core.function import Reshape

        # if shape is tuple of tuple, convert to tuple
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]

        return Reshape()(self, shape=shape)

    def transpose(self, *transpose: Optional[int | tuple]):
        from src.main.ragnarok.core.function import Transpose

        if not transpose:
            return Transpose()(self)

        if len(transpose) == 1 and isinstance(transpose[0], tuple):
            transpose = transpose[0]

        return Transpose()(self, transpose=transpose)

    def sum(self, axis=None, keepdims=False):
        from src.main.ragnarok.core.function import Sum

        return Sum()(self, axis=axis, keepdims=keepdims)

    @property
    def T(self):
        return self.transpose()

    def backward(self, retain_grad=False, enable_double_backprop=False):
        if self._creator is None:
            raise VariableError(
                "The creator of this variable is None. backward propagation is not possible."
            )

        if self._grad is None:
            # initialize with first grad as 1
            self._grad = Variable(np.ones_like(self.data))

        # higher generation popped first
        function_queue = []
        # idx used to prevent crash in heapq operation: if gen is same, comparing between function will crash.
        idx = 0
        heapq.heappush(function_queue, ((-self.creator.gen, idx), self.creator))
        visited = {self.creator}
        idx += 1

        # DFS to iterate all related variables: use pop and append
        while function_queue:
            _, function = heapq.heappop(function_queue)

            doutputs = [output().grad for output in function.outputs]

            # For first-order differentiation, backpropagation of grad variable is not needed
            with using_config("enable_backprop", enable_double_backprop):
                dinputs = function.backward(*doutputs)
                if not isinstance(dinputs, tuple):
                    dinputs = (dinputs,)

                inputs = function.inputs

                for input_, dinput in zip(inputs, dinputs):
                    if input_.grad is not None:
                        # For the function has more than one input and same inputs are used for the function
                        # e.g. Add()(x, x)
                        input_.grad = input_.grad + dinput
                    else:
                        input_.grad = dinput

                    if input_.creator is not None:
                        next_creator = input_.creator
                        if next_creator in visited:
                            continue
                        heapq.heappush(
                            function_queue, ((-next_creator.gen, idx), next_creator)
                        )
                        visited.add(next_creator)
                        idx += 1
            if not retain_grad:
                # In general case, current output grad is not needed anymore
                for output in function.outputs:
                    output().grad = None


class VariableError(RuntimeError):
    pass


def to_variable(x: int | float | np.ndarray | np.generic | Variable) -> Variable:
    if isinstance(x, Variable):
        return x
    return Variable(x)


def ones_like(x: Variable) -> Variable:
    return Variable(np.ones_like(x.data, dtype=x.dtype))


def zeros(shape, dtype=np.float32):
    return Variable(np.zeros(shape, dtype=dtype))
