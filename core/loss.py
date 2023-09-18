import numpy as np


def cross_entropy(y: np.ndarray, t: np.ndarray) -> np.float64:
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    h = 1e-7
    return -np.sum(t * np.log(y + h)) / batch_size
