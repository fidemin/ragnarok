import numpy as np


class Variable:
    def __init__(self, data: int | float | np.ndarray | np.generic):
        """
        Args:
            data: numpy array or int or float
        """

        if not (isinstance(data, (int, float, np.ndarray, np.generic))):
            raise VariableError('data should be numpy array or int or float')

        self._data = data

    @property
    def data(self) -> int | float | np.ndarray | np.generic:
        return self._data


class VariableError(RuntimeError):
    pass
