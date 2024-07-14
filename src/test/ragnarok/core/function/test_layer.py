import pytest

from src.main.ragnarok.core.function.layer import Linear
from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, ones_like


class TestLayer:
    @pytest.mark.parametrize(
        "x, W, b, expected",
        [
            (
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # 3 X 2
                [[1.0, 2.0], [2.0, 1.0]],  # 2 X 2
                [1.0, 3.0],  # (2,)
                [[6.0, 7.0], [9.0, 10.0], [12.0, 13.0]],  # 3 X 2
            ),
        ],
    )
    def test_forward(self, x, W, b, expected):
        x_var = Variable(x)
        W_var = Variable(W)
        b_var = Variable(b)
        expected_var = Variable(expected)

        f = Linear()
        actual = f(x_var, W_var, b_var)

        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "x, W, b, expected_dx, expected_dW, expected_db",
        [
            (
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # x 3 X 2
                [[1.0, 2.0], [3.0, 1.0]],  # W 2 X 2
                [1.0, 3.0],  # b (2,)
                [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]],  # dx 3 X 2
                [[6.0, 6.0], [9.0, 9.0]],  # dW 2 X 2
                [3.0, 3.0],  # db (2,)
            ),
        ],
    )
    def test_backward(self, x, W, b, expected_dx, expected_dW, expected_db):
        x_var = Variable(x)
        W_var = Variable(W)
        b_var = Variable(b)

        f = Linear()
        f(x_var, W_var, b_var)
        actual_dx, actual_dW, actual_db = f.backward(
            Variable([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        )

        assert allclose(actual_dx, Variable(expected_dx))
        assert allclose(actual_dW, Variable(expected_dW))
        assert allclose(actual_db, Variable(expected_db))

    @pytest.mark.parametrize(
        "x, W, b",
        [
            (
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # 3 X 2
                [[1.0, 2.0], [3.0, 1.0]],  # 2 X 2
                [1.0, 3.0],  # (2,)
            ),
        ],
    )
    def test_gradient_check(self, x, W, b):
        x_var = Variable(x)
        W_var = Variable(W)
        b_var = Variable(b)

        f = Linear()
        temp = f(x_var, W_var, b_var)

        actual = f.backward(ones_like(temp))

        expected = numerical_diff(f, x_var, W_var, b_var)

        assert allclose(actual[0], expected[0])  # dx
        assert allclose(actual[1], expected[1])  # dW
        assert allclose(actual[2], expected[2])  # db
