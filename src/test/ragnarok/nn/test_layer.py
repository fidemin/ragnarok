import pytest

from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.core.variable.dtype import float32
from src.main.ragnarok.nn.core.layer import Affine
from src.main.ragnarok.nn.core.parameter import Parameter


class TestAffine:
    @pytest.mark.parametrize(
        "use_bias",
        [None, True, False],
        ids=["default_bias", "with_bias", "without_bias"],
    )
    def test_init_use_bias(self, use_bias):
        if use_bias is None:
            use_bias = True  # default value
            affine = Affine(out_size=3)
        else:
            affine = Affine(out_size=3, use_bias=use_bias)

        assert affine.use_bias == use_bias

    def test_init_without_in_size(self):
        affine = Affine(out_size=3)
        assert affine.in_size is None
        assert affine.use_bias is True
        assert affine.dtype == float32
        assert affine.params == {}

    def test_init_with_in_size(self):
        affine = Affine(out_size=3, in_size=2)
        assert affine.in_size == 2
        assert affine.use_bias is True
        assert affine.dtype == float32
        assert "W" in affine.params
        assert "b" in affine.params
        assert affine.params["W"].shape == (2, 3)
        assert affine.params["b"].shape == (3,)

    @pytest.mark.parametrize(
        "use_bias, x, W, b, expected",
        [
            (
                True,
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # 3 X 2
                [[1.0, 2.0], [2.0, 1.0]],  # 2 X 2
                [1.0, 3.0],  # (2,)
                [[6.0, 7.0], [9.0, 10.0], [12.0, 13.0]],  # 3 X 2
            ),
            (
                False,
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],  # 3 X 2
                [[1.0, 2.0], [2.0, 1.0]],  # 2 X 2
                [1.0, 3.0],  # (2,)
                [[5.0, 4.0], [8.0, 7.0], [11.0, 10.0]],  # 3 X 2
            ),
        ],
    )
    def test_forward(self, use_bias, x, W, b, expected):
        # Given
        param_W = Parameter(W, name="W")
        affine = Affine(
            out_size=param_W.shape[-1], in_size=param_W.shape[0], use_bias=use_bias
        )
        affine.params["W"] = Parameter(W)
        if use_bias:
            affine.params["b"] = Parameter(b)

        x = Variable(x)

        # When
        y = affine.forward(x)
        assert y.shape == (3, 2)
        assert allclose(y, Variable(expected))

    @pytest.mark.parametrize(
        "x, expected_shape",
        [
            ([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], (3, 3)),
            ([[[1.0, 2.0], [2.0, 3.0]]], (1, 2, 3)),
        ],
    )
    def test_forward_lazy_init(self, x, expected_shape):
        # Given
        affine = Affine(out_size=3)
        x = Variable(x)

        # When
        y = affine.forward(x)

        # Then
        assert y.shape == expected_shape
