import numpy as np


def gradient(func, x: np.ndarray) -> np.ndarray:
    """
    :param func: function for gradient calculation. argument should be ndarray
    :param x: position for gradient
    :return: gradient at the position
    """
    h = 1e-4

    f_for_plus_h = func(x + h)
    f_for_minus_h = func(x - h)

    return (f_for_plus_h - f_for_minus_h) / (2 * h)



