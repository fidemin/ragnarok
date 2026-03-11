import numpy as np
import pytest

from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose, numerical_diff
from ragnarok.nn.function.loss import (
    MeanSquaredError,
    CrossEntropyError,
    SoftMaxLoss,
)


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
        x0_var = Tensor(x0)
        x1_var = Tensor(x1)
        expected_var = Tensor(expected)

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
        x0_var = Tensor(x0)
        x1_var = Tensor(x1)

        f = MeanSquaredError()
        f(x0_var, x1_var)
        actual_dx0, actual_dx1 = f.backward(Tensor(1.0))
        assert allclose(actual_dx0, Tensor(expected_x0))
        assert allclose(actual_dx1, Tensor(expected_x1))

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
        x0 = Tensor(x0)
        x1 = Tensor(x1)

        f = MeanSquaredError()
        f(x0, x1)

        actual_dx0, actual_dx1 = f.backward(Tensor(1.0))
        expected_dx0, expected_dx1 = numerical_diff(f, x0, x1)

        assert actual_dx0.shape == expected_dx0.shape
        assert actual_dx1.shape == expected_dx1.shape
        assert allclose(actual_dx0, expected_dx0)
        assert allclose(actual_dx1, expected_dx1)


class TestCrossEntropyError:
    @pytest.mark.parametrize(
        "test_y, test_t, expected",
        [
            ([0.1, 0.9], [0.0, 1.0], -np.log(0.9)),
            (
                [[0.1, 0.9], [0.3, 0.7]],
                [[0.0, 1.0], [1.0, 0.0]],
                (-np.log(0.9) + -np.log(0.3)) / 2.0,
            ),
        ],
    )
    def test_forward(self, test_y, test_t, expected):
        y = Tensor(test_y)
        t = Tensor(test_t)
        f = CrossEntropyError()
        actual = f(y, t)
        assert allclose(actual, Tensor(expected))

    @pytest.mark.parametrize(
        "test_y, test_t, expected_dy, expected_dt",
        [
            (
                [0.1, 0.9],
                [0.0, 1.0],
                [0.0, -1.11111099 * 2.0],
                [2.30258409 * 2.0, 0.1053604 * 2.0],
            ),
            (
                [[0.1, 0.9], [0.3, 0.7]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, -0.5555555 * 2.0], [-1.66666617 * 2.0, 0.0]],
                [
                    [1.15129205 * 2.0, 0.0526802 * 2.0],
                    [0.60198624 * 2.0, 0.1783374 * 2.0],
                ],
            ),
        ],
    )
    def test_backward(self, test_y, test_t, expected_dy, expected_dt):
        y = Tensor(test_y)
        t = Tensor(test_t)
        f = CrossEntropyError()
        for_weak_ref = f(y, t)
        actual_dy, actual_dt = f.backward(Tensor(2.0))

        expected_dy = Tensor(expected_dy)
        expected_dt = Tensor(expected_dt)
        assert allclose(actual_dy, expected_dy)
        assert allclose(actual_dt, expected_dt)

    @pytest.mark.parametrize(
        "test_y, test_t",
        [
            ([0.1, 0.9], [0.0, 1.0]),
            ([[0.1, 0.9], [0.3, 0.7]], [[0.0, 1.0], [1.0, 0.0]]),
        ],
    )
    def test_gradient_check(self, test_y, test_t):
        f = CrossEntropyError()
        y = Tensor(test_y)
        t = Tensor(test_t)
        for_weak_ref = f(y, t)
        actual_dy, actual_dt = f.backward(Tensor(1.0))
        expected_dy, expected_dt = numerical_diff(f, y, t)
        assert allclose(actual_dy, expected_dy)
        assert allclose(actual_dt, expected_dt)


class TestSoftMaxLoss:
    @pytest.mark.parametrize(
        "y, t, expected",
        [
            (
                [2.0, 1.0],
                [0.4, 0.6],
                -0.4 * np.log(0.731059) + -0.6 * np.log(0.268941),
            ),
            (
                [[2.0, 1.0], [1.0, 2.0]],
                [[0.4, 0.6], [0.0, 1.0]],
                (-0.4 * np.log(0.731059) + -0.6 * np.log(0.268941) + -np.log(0.731059))
                / 2,
            ),
        ],
    )
    def test_forward(self, y, t, expected):
        y = Tensor(y)
        t = Tensor(t)
        f = SoftMaxLoss()
        actual = f(y, t)
        assert allclose(actual, Tensor(expected))

    @pytest.mark.parametrize(
        "y, t",
        [
            (
                [2.0, 1.0],
                [0.4, 0.6],
            ),
            (
                [[2.0, 1.0], [1.0, 2.0]],
                [[0.4, 0.6], [0.0, 1.0]],
            ),
        ],
    )
    def test_gradient_check(self, y, t):
        f = SoftMaxLoss()
        y = Tensor(y)
        t = Tensor(t)
        for_weak_ref = f(y, t)
        actual_dy, actual_dt = f.backward(Tensor(1.0))
        expected_dy, expected_dt = numerical_diff(f, y, t)
        assert allclose(actual_dy, expected_dy)
        assert allclose(actual_dt, expected_dt)
