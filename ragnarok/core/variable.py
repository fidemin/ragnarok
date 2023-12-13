import numpy as np


class Variable:
    def __init__(self, data: int | float | np.ndarray | np.generic, creator=None):
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

        self._data = data
        self._creator = creator
        self._grad = None
        self._gen = 0

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

    def backward(self):
        if self._creator is None:
            raise VariableError('The creator of this variable is None. backward propagation is not possible.')

        if self._grad is None:
            self._grad = Variable(np.ones_like(self.data))

        # initial value
        functions = [self._creator]

        # DFS to iterate all related variables: use pop and append
        while functions:
            function = functions.pop()
            doutputs = []

            skip_backward = False
            for output in function.outputs:
                if output.grad is None:
                    # All of outputs of the function has not been processed yet.
                    skip_backward = True
                    break
                doutputs.append(output.grad)

            if skip_backward:
                continue

            dinputs = function.backward(*doutputs)
            if not isinstance(dinputs, tuple):
                dinputs = (dinputs,)

            inputs = function.inputs

            for input_, dinput in zip(inputs, dinputs):
                if input_.grad is not None:
                    input_.grad = Variable(input_.grad.data + dinput.data)
                else:
                    input_.grad = dinput

                if input_.creator is not None:
                    next_creator = input_.creator
                    functions.append(next_creator)


class VariableError(RuntimeError):
    pass
