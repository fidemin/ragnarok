from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import Sigmoid, sigmoid
from src.main.ragnarok.nn.layer.activation import ReLU


class TestSigmoid:
    def test_forward(self, mocker):
        # Given
        layer = Sigmoid()
        x = Variable([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, sigmoid(x))


class TestReLU:
    def test_forward(self, mocker):
        # Given
        layer = ReLU()
        x = Variable([-1.0, 0.0, 1.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, Variable([0.0, 0.0, 1.0]))
