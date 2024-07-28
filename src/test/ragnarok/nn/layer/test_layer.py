import pytest

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.layer.layer import Layer


class MockLayer(Layer):
    def _forward(self, *variables: Variable, **kwargs):
        return variables[0]


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
