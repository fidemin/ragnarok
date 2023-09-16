import numpy as np


def gradient(func, x: np.ndarray) -> np.ndarray:
    """
    calculate numerical gradient at position x for function

    :param func: function for gradient calculation. argument should be ndarray
    :param x: position for gradient
    :return: gradient at the position
    """
    h = 1e-4

    f_for_plus_h = func(x + h)
    f_for_minus_h = func(x - h)

    return (f_for_plus_h - f_for_minus_h) / (2 * h)


def gradient_descent(func, init_x: np.ndarray, lr=0.01, n_iter=100) -> np.ndarray:
    """
    calculate gradient descent from initial position for function

    :param func: function for gradient descent calculation. argument should be ndarray
    :param init_x: initial position of x
    :param lr: learning rate
    :param n_iter: number of iterations
    :return: the position of x after gradient descent calculation
    """
    x = init_x
    for i in range(n_iter):
        grad = gradient(func, x)
        x -= grad * lr

    return x
