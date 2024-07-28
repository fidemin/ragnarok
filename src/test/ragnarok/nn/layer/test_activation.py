from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import Sigmoid, sigmoid


class TestSigmoid:
    def test_forward(self, mocker):
        # Given
        layer = Sigmoid()
        x = Variable([1.0, 2.0, 3.0])

        # When
        y = layer.forward(x)

        # Then
        assert allclose(y, sigmoid(x))
