import pytest

from src.main.ragnarok.graph.node import VariableNode, FunctionNode


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
    def test_draw_verbose_true(self, id_, name, shape, dtype, expected):
        kwargs = {}
        if name is not None:
            kwargs['name'] = name
        if shape is not None:
            kwargs['shape'] = shape
        if dtype is not None:
            kwargs['dtype'] = dtype
        node = VariableNode(id_, **kwargs)
        assert node.draw(verbose=True) == expected


class TestFunctionNode:
    @pytest.mark.parametrize('id_, name, input_ids, output_ids, expected', [
        (0, 'Square', [], [], '0 [label="Square", color=lightblue, style=filled, shape=box]'),
    ])
    def test_to_str(self, id_, name, input_ids, output_ids, expected):
        node = FunctionNode(id_, name, input_ids=input_ids, output_ids=output_ids)
        assert node.to_str() == expected

    @pytest.mark.parametrize('id_, name, input_ids, output_ids, expected_list', [
        (0, 'Square', [1, 2], [3, 4],
         ['0 [label="Square", color=lightblue, style=filled, shape=box]', '1 -> 0', '2 -> 0', '0 -> 3', '0 -> 4']),
    ])
    def test_draw(self, id_, name, input_ids, output_ids, expected_list):
        node = FunctionNode(id_, name, input_ids=input_ids, output_ids=output_ids)
        assert node.draw() == '\n'.join(expected_list)
