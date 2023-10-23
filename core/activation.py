import numpy as np


def step(x: np.ndarray) -> np.ndarray:
    """
    step function

    :param x: ndarray of numbers e.g. [1.0, 0.0, 5.0]
    :return: returns the result of step function as ndarray
    """
    y = x > 0
    return y.astype(np.int_)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    sigmoid function

    :param x: ndarray of numbers e.g. [1.0, 0.0, 5.0]
    :return: returns the result of sigmoid function as ndarray
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))


def softmax(x: np.ndarray) -> np.ndarray:
    """
    softmax function

    :param x: ndarray of numbers with dim >= 1 e.g. [[1.0, 0.0, 5.0], [1.0, 2.0, 2.0]]
    :return: returns the result of softmax function as ndarray. shape is same as x
    """
    dim_is_one = (x.ndim == 1)
    original_shape = x.shape

    if dim_is_one:
        x = x.reshape(1, -1)
    max_value = np.max(x, axis=1)

    # To prevent overflow of e(x), change the all input value to <0 value.
    x_normal = (x.T - max_value).T

    exp_ = np.exp(x_normal)
    exp_sum = np.sum(exp_, axis=1)

    result = (exp_.T / exp_sum).T

    if dim_is_one:
        return result.reshape(original_shape)

    return result
