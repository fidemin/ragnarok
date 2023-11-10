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

        self._data = data
        self._creator = creator
        self._grad = None

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


class VariableError(RuntimeError):
    pass
