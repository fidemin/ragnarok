import numpy as np

from core import layer, numerical, activation


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
