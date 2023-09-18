import numpy as np

from core.activation import sigmoid
from core.layer import Layer


def func1(x: np.ndarray) -> np.ndarray:
    return np.power(x, 2)


def test_predict():
    layer = Layer(2, 3)
    predict = layer.predict(np.array([1.0, 2.0]), sigmoid)
    assert predict.shape[0] == 3


def test_gradient():
    layer = Layer(2, 3)
    layer.W = np.array([[3.0, 4.0], [0.0, 2.0], [5.0, 0.0]])
    layer.b = np.array([1.0, 2.0, 0.0])

    expected_W_grad = np.array([[6.0, 8.0], [0.0, 4.0], [10.0, 0.0]])
    expected_b_grad = np.array([2.0, 4.0, 0.0])

    layer.gradient(func1)
    assert np.allclose(layer.W_grad, expected_W_grad)
    assert np.allclose(layer.b_grad, expected_b_grad)
