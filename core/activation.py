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


def softmax(x: np.ndarray) -> np.ndarray:
    """
    softmax function

    :param x: ndarray of numbers e.g. [1.0, 0.0, 5.0]
    :return: returns the result of softmax function as ndarray
    """
    max_value = np.max(x)

    # To prevent overflow of e(x), change the all input value to <0 value.
    x_normal = x - max_value

    exp_ = np.exp(x_normal)
    exp_sum = np.sum(exp_)

    return exp_ / exp_sum
