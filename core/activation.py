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

    # to prevent overflow of exponential with float64
    # if x < -20.0, sigmoid is 0.0 in float64 with 1e-8 precision
    max_exp_arg = 20.0
    indexes = np.where(x < -max_exp_arg)
    x[indexes] = -max_exp_arg
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    # to prevent overflow of exponential with float64
    # if x < -20.0, tanh is -1.0 in float64 with 1e-8 precision
    # if x > 20.0, tanh is 1.0 in float64 with 1e-8 precision
    max_exp_arg = 20.0
    indexes_low = np.where(x < -max_exp_arg)
    x[indexes_low] = -max_exp_arg
    indexes_high = np.where(x > max_exp_arg)
    x[indexes_high] = max_exp_arg
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


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
