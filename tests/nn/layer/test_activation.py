from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose
from ragnarok.nn.function.activation import sigmoid, tanh, relu
from ragnarok.nn.layer.activation import ReLU, Tanh, Sigmoid


class TestSigmoid:
    def test_forward(self):
        # Given
        layer = Sigmoid()
        x = Tensor([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, sigmoid(x))


class TestTanh:
    def test_forward(self):
        # Given
        layer = Tanh()
        x = Tensor([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, tanh(x))


class TestReLU:
    def test_forward(self):
        # Given
        layer = ReLU()
        x = Tensor([-1.0, 0.0, 1.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, relu(x))
