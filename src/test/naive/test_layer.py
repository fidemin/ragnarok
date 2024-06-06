import numpy as np

from src.main.core.activation import sigmoid
from src.main.naive.layer import Layer


def func1(x: np.ndarray) -> np.ndarray:
    return np.sum(np.power(x, 2))


def test_predict():
    layer = Layer(2, 3)
    predict = layer.predict(np.array([1.0, 2.0]), sigmoid)
    assert predict.shape[0] == 3


def test_gradient():
    layer = Layer(2, 3)
    layer.W = np.array([[3.0, 4.0, 2.0], [0.0, 2.0, 5.0]])
    layer.b = np.array([1.0, 2.0, 0.0])

    expected_W_grad = np.array([[6.0, 8.0, 4.0], [0.0, 4.0, 10.0]])
    expected_b_grad = np.array([2.0, 4.0, 0.0])

    layer.gradient(func1)
    assert np.allclose(layer.W_grad, expected_W_grad)
    assert np.allclose(layer.b_grad, expected_b_grad)


def test_update_params_from_gradient_descent():
    layer = Layer(2, 3)
    layer.W = np.array([[3.0, 4.0, 2.0], [0.0, 2.0, 5.0]])
    layer.b = np.array([1.0, 2.0, 0.0])

    layer.W_grad = np.array([[6.0, 8.0, 4.0], [0.0, 4.0, 10.0]])
    layer.b_grad = np.array([2.0, 4.0, 0.0])

    layer.gradient(loss_func=func1)
    layer.update_params_from_gradient_descent(lr=0.1)

    expected_W = [[2.4, 3.2, 1.6], [0.0, 1.6, 4.0]]
    expected_b = [0.8, 1.6, 0.0]

    assert np.allclose(layer.W, expected_W)
    assert np.allclose(layer.b, expected_b)
