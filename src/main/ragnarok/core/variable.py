import heapq

import numpy as np


class Variable:
    __array_priority__ = 200

    def __init__(self, data: int | float | np.ndarray | np.generic, name=None):
        """
        Args:
            data: numpy array or int or float
        Attributes:
            data: original data
            creator: Function instance which generate this variable
        """

        if not (isinstance(data, (int, float, np.ndarray, np.generic))):
            raise VariableError('data should be numpy array or int or float')

        if isinstance(data, (int, float)):
            data = np.array(data)

        self._name = name
        self._data = data
        self._creator = None
        self._grad = None
        self._gen = 0

    def __len__(self):
        return self._data.shape[0] if self._data.shape else 0

    def __repr__(self):
        return f'Variable({str(self._data)})'

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

    def set_creator(self, creator):
        self._creator = creator
        self._gen = creator.gen + 1

    @property
    def data(self) -> int | float | np.ndarray | np.generic:
        return self._data

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
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def dtype(self):
        return self._data.dtype.name

    def backward(self, retain_grad=False):
        if self._creator is None:
            raise VariableError('The creator of this variable is None. backward propagation is not possible.')

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

            dinputs = function.backward(*doutputs)
            if not isinstance(dinputs, tuple):
                dinputs = (dinputs,)

            inputs = function.inputs

            for input_, dinput in zip(inputs, dinputs):
                if input_.grad is not None:
                    # For the function has more than one input and same inputs are used for the function
                    # e.g. Add()(x, x)
                    input_.grad = Variable(input_.grad.data + dinput.data)
                else:
                    input_.grad = dinput

                if input_.creator is not None:
                    next_creator = input_.creator
                    if next_creator in visited:
                        continue
                    heapq.heappush(function_queue, ((-next_creator.gen, idx), next_creator))
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
