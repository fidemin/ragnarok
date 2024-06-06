import numpy as np

from src.main.core import layer
from src.main.core import activation, numerical


class TestRelu:
    def test_forward(self):
        relu = layer.Relu()
        x = np.array([[1.0, -1.0, 2.0], [0.0, 1.0, -10.0]])
        actual = relu.forward(x)
        expected = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]])

        assert np.all(actual == expected)

    def test_backward(self):
        relu = layer.Relu()
        x = np.array([[1.0, -1.0, 2.0], [0.0, 1.0, -10.0]])
        relu.forward(x)

        dout = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

        actual = relu.backward(dout)
        expected = np.array([[3.0, 0.0, 5.0], [0.0, 7.0, 0.0]])

        assert np.all(actual == expected)


class TestSigmoid:
    def test_forward(self):
        sigmoid = layer.Sigmoid()
        x = np.array([[10.0, 0.0, -1.5]])

        actual = sigmoid.forward(x)
        expected = np.array([[0.9999546021312978, 0.5, 0.18242552380627775]])

        assert np.allclose(actual, expected)

    def test_backward(self):
        sigmoid_layer = layer.Sigmoid()
        x = np.array([[10.0, 0.0, -1.5]])

        sigmoid_layer.forward(x)

        dout = np.array([[1.0, 1.0, 1.0]])

        actual = sigmoid_layer.backward(dout)
        # compare it with numerical value
        expected = numerical.gradient(activation.sigmoid, x)

        assert np.allclose(actual, expected)


class TestBatchNorm:
    def test_forward(self):
        x = np.array([
            [[10.0, 0.0], [1.0, 2.0]],
            [[2.0, 4.0], [3.0, 8.0]],
            [[3.0, 6.0], [1.0, 4.0]],
        ])

        gamma = np.ones_like(x[0]) + 1
        beta = np.zeros_like(x[0])
        batch_norm = layer.BatchNorm(gamma, beta)

        actual = batch_norm.forward(x)

        expected = np.array([
            [[2.80975743, -2.67261242], [-1.41421355, -2.13808993]],
            [[-1.68585446, 0.53452248], [2.82842711, 2.67261242]],
            [[-1.12390297, 2.13808993], [-1.41421355, -0.53452248]]
        ])

        assert np.allclose(actual, expected)

    def test_backward(self):
        x = np.array([
            [[10.0, 0.0], [1.0, 2.0]],
            [[2.0, 4.0], [3.0, 8.0]],
            [[3.0, 6.0], [1.0, 4.0]],
        ])
        batch_norm = layer.BatchNorm.from_shape(x.shape[1:])

        batch_norm.forward(x)
        dout = np.ones(x.shape)

        batch_norm.backward(dout)
