import numpy as np


class Variable:
    def __init__(self, data: int | float | np.ndarray):
        """
        Args:
            data: numpy array or int or float
        """

        if not (isinstance(data, int) or isinstance(data, float) or isinstance(data, np.ndarray)):
            raise VariableError('data should be numpy array or int or float')

        self._data = data

    @property
    def data(self) -> int | float | np.ndarray:
        return self._data


class VariableError(RuntimeError):
    pass
