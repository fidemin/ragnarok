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


def log_loss(y: np.ndarray, t: np.ndarray) -> np.float64:
    """
    :param y: input. ndarray of probability. 0.0 <= value <= 1.0
    :param t: target. ndarray of 1 or 0
    :return: total sum of loss
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    h = 1e-7
    return -np.sum(t * np.log(y + h) + (1 - t) * np.log(1 - y + h)) / batch_size


def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
