import numpy as np


def cross_entropy(y: np.ndarray, t: np.ndarray) -> np.float64:
    """
    :param y: input
    :param t: target with one-hot encoding format. t should have the same dimension with input.
    :return: total sum of loss
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    h = 1e-7
    return -np.sum(t * np.log(y + h)) / batch_size
