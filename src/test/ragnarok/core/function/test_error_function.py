import pytest

from src.main.ragnarok.core.function.error_function import MeanSquaredError
from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable


class TestMeanSquaredFunction:
    @pytest.mark.parametrize(
        "x0, x1, expected",
        [
            (1.0, 1.0, 0),
            ([1.0, 2.0], [1.0, 2.0], 0.0),
            ([1.0, 2.0], [2.0, 1.0], 1.0),
            ([[1.0, 2.0], [2.0, 1.0]], [[1.0, 2.0], [0.0, 3.0]], 4.0),
        ],
    )
    def test_forward(self, x0, x1, expected):
        x0_var = Variable(x0)
        x1_var = Variable(x1)
        expected_var = Variable(expected)

        f = MeanSquaredError()
        actual = f(x0_var, x1_var)

        assert actual.shape == expected_var.shape
        assert allclose(actual, expected_var)

    @pytest.mark.parametrize(
        "x0, x1, expected_x0, expected_x1",
        [
            (1.0, 1.0, 0, 0),
            ([1.0, 2.0], [1.0, 2.0], [0.0, 0.0], [0.0, 0.0]),
            ([1.0, 2.0], [2.0, 1.0], [-1.0, 1.0], [1.0, -1.0]),
            (
                [[1.0, 2.0], [2.0, 1.0]],
                [[1.0, 2.0], [0.0, 3.0]],
                [[0.0, 0.0], [2.0, -2.0]],
                [[0.0, 0.0], [-2.0, 2.0]],
            ),
        ],
    )
    def test_backward(self, x0, x1, expected_x0, expected_x1):
        x0_var = Variable(x0)
        x1_var = Variable(x1)

        f = MeanSquaredError()
        f(x0_var, x1_var)
        actual_dx0, actual_dx1 = f.backward(Variable(1.0))
        assert allclose(actual_dx0, Variable(expected_x0))
        assert allclose(actual_dx1, Variable(expected_x1))

    @pytest.mark.parametrize(
        "x0, x1",
        [
            (1.0, 1.0),
            ([1.0, 2.0], [1.0, 2.0]),
            ([1.0, 2.0], [2.0, 1.0]),
            (
                [[1.0, 2.0], [2.0, 1.0]],
                [[1.0, 2.0], [0.0, 3.0]],
            ),
        ],
    )
    def test_gradient_check(self, x0, x1):
        x0 = Variable(x0)
        x1 = Variable(x1)

        f = MeanSquaredError()
        f(x0, x1)

        actual_dx0, actual_dx1 = f.backward(Variable(1.0))
        expected_dx0, expected_dx1 = numerical_diff(f, x0, x1)

        assert actual_dx0.shape == expected_dx0.shape
        assert actual_dx1.shape == expected_dx1.shape
        assert allclose(actual_dx0, expected_dx0)
        assert allclose(actual_dx1, expected_dx1)
