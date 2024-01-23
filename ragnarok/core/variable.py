import heapq
import weakref

import numpy as np


class Variable:
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

    def set_creator(self, creator):
        self._creator = weakref.ref(creator)
        self._gen = creator.gen + 1

    @property
    def data(self) -> int | float | np.ndarray | np.generic:
        return self._data

    @property
    def creator(self):
        if self._creator is not None:
            return self._creator()
        return None

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

    def backward(self):
        if self._creator is None:
            raise VariableError('The creator of this variable is None. backward propagation is not possible.')

        if self._grad is None:
            self._grad = Variable(np.ones_like(self.data))

        # higher generation popped first
        function_queue = []
        # idx used to prevent crash in heapq operation: if gen is same, comparing between function will crash.
        idx = 0
        heapq.heappush(function_queue, (-self.creator.gen, idx, self._creator()))
        visited = {self._creator()}
        idx += 1

        # DFS to iterate all related variables: use pop and append
        while function_queue:
            _, _, function = heapq.heappop(function_queue)

            doutputs = [output.grad for output in function.outputs]

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
                    heapq.heappush(function_queue, (-next_creator.gen, idx, next_creator))
                    visited.add(next_creator)
                    idx += 1


class VariableError(RuntimeError):
    pass
