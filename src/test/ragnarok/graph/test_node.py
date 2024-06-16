import pytest

from src.main.ragnarok.graph.node import VariableNode


class TestVariableNode:
    @pytest.mark.parametrize('id_, name, shape, dtype, expected', [
        (0, None, None, None, '0 [label="", color=orange, style=filled]'),
        (0, 'x', None, None, '0 [label="x", color=orange, style=filled]'),
        (0, None, (2, 3), None, '0 [label="(2, 3)", color=orange, style=filled]'),
        (0, None, None, 'float32', '0 [label="float32", color=orange, style=filled]'),
        (0, 'x', (2, 3), None, '0 [label="x: (2, 3)", color=orange, style=filled]'),
        (0, 'x', None, 'float32', '0 [label="x: float32", color=orange, style=filled]'),
        (0, None, (2, 3), 'float32', '0 [label="(2, 3) float32", color=orange, style=filled]'),
        (0, 'x', (2, 3), 'float32', '0 [label="x: (2, 3) float32", color=orange, style=filled]'),
    ])
    def test_to_str_verbose_true(self, id_, name, shape, dtype, expected):
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if shape is not None:
            kwargs['shape'] = shape
        if dtype is not None:
            kwargs['dtype'] = dtype
        node = VariableNode(id_, **kwargs)
        assert node.to_str(verbose=True) == expected
