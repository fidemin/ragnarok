import pytest

from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.layer.layer import Layer


class MockLayer(Layer):
    def _forward(self, *variables: Variable, **kwargs):
        return variables


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
        "variables",
        [
            [Variable([1, 2, 3])],
            [Variable([1, 2, 3]), Variable([4, 5, 6])],
        ],
    )
    def test_forward(self, variables):
        layer = MockLayer()

        # y can be a single Variable or a list of Variables
        y = layer.forward(variables)

        if not isinstance(y, list):
            y = [y]

        for i, var in enumerate(variables):
            assert allclose(y[i], var)
