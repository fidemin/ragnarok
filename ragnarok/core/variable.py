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

    @property
    def data(self) -> int | float | np.ndarray | np.generic:
        return self._data

    @property
    def creator(self):
        return self._creator


class VariableError(RuntimeError):
    pass
