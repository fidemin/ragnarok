from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import sigmoid, tanh, relu
from src.main.ragnarok.nn.layer.activation import ReLU, Tanh, Sigmoid


class TestSigmoid:
    def test_forward(self):
        # Given
        layer = Sigmoid()
        x = Variable([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)[0]

        # Then
        assert allclose(y, sigmoid(x))


class TestTanh:
    def test_forward(self):
        # Given
        layer = Tanh()
        x = Variable([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)[0]

        # Then
        assert allclose(y, tanh(x))


class TestReLU:
    def test_forward(self):
        # Given
        layer = ReLU()
        x = Variable([-1.0, 0.0, 1.0])

        # When
        y = layer.forward(x)[0]

        # Then
        assert allclose(y, relu(x))
