import numpy as np
import pytest

from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, ones_like
from src.main.ragnarok.nn.function.activation import Sigmoid, Tanh, ReLU, Softmax


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


class TestTanh:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([np.tanh(0.1), np.tanh(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Tanh()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f = Tanh()
        y_for_weak_ref = f(test_input)
        dout = Variable(np.array([1.0, 1.0]))

        expected = Variable(np.array([1 - np.tanh(0.1) ** 2, 1 - np.tanh(0.2) ** 2]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Tanh()
        test_input = Variable(np.array([[0.1, 0.2]]))

        y_for_weak_ref = f(test_input)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestReLU:
    def test_forward(self):
        x = Variable([1.0, -1.0])
        expected = Variable([1.0, 0.0])

        f = ReLU()
        actual = f.forward(x)

        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    def test_backward(self):
        x = Variable([2.0, -1.0])

        f = ReLU()
        for_weak_ref = f(x)
        actual = f.backward(ones_like(x))

        expected = Variable([1.0, 0.0])
        assert allclose(actual, expected)

    def test_gradient_check(self):
        x = Variable([2.0, -1.0])

        f = ReLU()
        for_weak_ref = f(x)
        actual = f.backward(ones_like(x))

        expected = numerical_diff(f, x)

        assert allclose(actual, expected)


class TestSoftmax:
    @pytest.mark.parametrize(
        "input_arr, expected_arr",
        [
            (
                [[0.1, -0.5, 0.2], [-0.2, -0.5, -0.8]],
                [
                    [0.37679223, 0.20678796, 0.41641981],
                    [0.43675182, 0.3235537, 0.23969448],
                ],
            ),
            # 1d array test
            (
                [0.1, -0.5, 0.2],
                [0.37679223, 0.20678796, 0.41641981],
            ),
        ],
    )
    def test_forward(self, input_arr, expected_arr):
        x = Variable(input_arr)
        expected = Variable(expected_arr)

        f = Softmax()
        actual = f.forward(x)

        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "input_arr",
        [
            [[0.4, 0.5, 0.2], [-0.2, -0.5, -0.8]],
            # 1d array test
            [0.1, -0.5, 0.2],
        ],
    )
    def test_gradient_check(self, input_arr):
        x = Variable(input_arr)

        f = Softmax()
        for_weak_ref = f(x)
        actual = f.backward(ones_like(x))

        expected = numerical_diff(f, x)

        assert allclose(actual, expected)
