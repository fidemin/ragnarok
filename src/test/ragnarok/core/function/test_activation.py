from src.main.ragnarok.core.function.activation import Sigmoid
from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, ones_like


class TestSigmoid:
    def test_forward(self):
        x = Variable([1.0, 2.0])
        expected = Variable([0.73105858, 0.88079708])

        f = Sigmoid()
        actual = f.forward(x)

        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    def test_backward(self):
        x = Variable([1.0, 2.0])

        f = Sigmoid()
        for_weak_ref = f(x)
        actual = f.backward(ones_like(x))

        expected = Variable([0.19661193, 0.10499359])
        assert allclose(actual, expected)

    def test_gradient_check(self):
        x = Variable([1.0, 2.0])

        f = Sigmoid()
        for_weak_ref = f(x)
        actual = f.backward(ones_like(x))

        expected = numerical_diff(f, x)

        assert allclose(actual, expected)
