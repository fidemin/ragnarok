import numpy as np


def step(x: np.ndarray) -> np.ndarray:
    """
    :param x: ndarray of numbers e.g. [1.0, 0.0, 5.0]
    :return: returns the result of step function as ndarray
    """
    y = x > 0
    return y.astype(np.int_)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
