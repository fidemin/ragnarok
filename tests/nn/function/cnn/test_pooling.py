import pytest

from ragnarok.core.tensor import Tensor, ones_like
from ragnarok.core.util import allclose, numerical_diff
from ragnarok.nn.function.cnn.pooling import MaxPooling


class TestPooling:
    @pytest.mark.parametrize(
        "test_input,kwargs,expected",
        [
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 1, "pool_w": 1, "stride": 1, "padding": 0},
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 0},
                Tensor([[[[5.0, 6.0], [8.0, 9.0]]]]),
            ),
            (
                Tensor([[[[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 1},
                Tensor(
                    [
                        [
                            [
                                [0.0, 2.0, 3.0, 3.0],
                                [4.0, 5.0, 6.0, 6.0],
                                [7.0, 8.0, 9.0, 9.0],
                                [7.0, 8.0, 9.0, 9.0],
                            ]
                        ]
                    ]
                ),
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 2, "padding": 1},
                Tensor([[[[1.0, 3.0], [7.0, 9.0]]]]),
            ),
        ],
    )
    def test_forward(self, test_input, kwargs, expected):
        max_pooling = MaxPooling()
        actual = max_pooling(test_input, **kwargs)
        assert actual.shape == expected.shape
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,kwargs,expected",
        [
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 1, "pool_w": 1, "stride": 1, "padding": 0},
                Tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]),
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 0},
                Tensor([[[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]]),
            ),
            (
                Tensor([[[[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 1},
                Tensor([[[[0.0, 1.0, 2.0], [1.0, 1.0, 2.0], [2.0, 2.0, 4.0]]]]),
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 2, "padding": 1},
                Tensor([[[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]]]),
            ),
        ],
    )
    def test_backward(self, test_input, kwargs, expected):
        max_pooling = MaxPooling()
        output = max_pooling(test_input, **kwargs)
        dout = ones_like(output)
        dx = max_pooling.backward(dout)
        assert dx.shape == expected.shape
        assert allclose(dx, expected)

    @pytest.mark.parametrize(
        "test_input,kwargs",
        [
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 1, "pool_w": 1, "stride": 1, "padding": 0},
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 0},
            ),
            (
                Tensor([[[[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 1, "padding": 1},
            ),
            (
                Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]]),
                {"pool_h": 2, "pool_w": 2, "stride": 2, "padding": 1},
            ),
        ],
    )
    def test_gradient_check(self, test_input, kwargs):
        f1 = MaxPooling()
        f2 = MaxPooling()

        output = f1(test_input, **kwargs)
        dout = ones_like(output)

        actual_dx = f1.backward(dout)
        expected_dx = numerical_diff(f2, test_input, **kwargs)

        assert allclose(actual_dx, expected_dx)
