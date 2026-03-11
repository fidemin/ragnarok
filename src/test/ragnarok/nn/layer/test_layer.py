import pytest

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.nn.layer.layer import Layer


class MockLayer(Layer):
    def _forward(self, *tensors: Tensor, **kwargs):
        return tensors


class TestLayer:
    @pytest.mark.parametrize("name", [None, "MockLayer"])
    def test_init(self, name):
        layer = MockLayer()
        if name is not None:
            layer = MockLayer(name=name)

        if name is not None:
            assert layer.name == name
        else:
            assert layer.name == "MockLayer"
        assert layer.params == {}

    @pytest.mark.parametrize(
        "tensors",
        [
            [Tensor([1, 2, 3])],
            [Tensor([1, 2, 3]), Tensor([4, 5, 6])],
        ],
    )
    def test_forward(self, tensors):
        layer = MockLayer()

        # y can be a single Tensor or a list of Variables
        y = layer.forward(tensors)

        if not isinstance(y, list):
            y = [y]

        for i, var in enumerate(tensors):
            assert allclose(y[i], var)
