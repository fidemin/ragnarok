import pytest

from src.main.ragnarok.graph.graph import Graph, InputVariableNode, InputFunctionNode


class TestGraph:
    @pytest.mark.parametrize('start_variable_id, variable_list, function_list, expected_list', [
        # no function
        (1, [InputVariableNode(1, 'x', (2, 3), 'float32', None)], [],
         ['1 [label="x: (2, 3) float32", color=orange, style=filled]']),
        # only one function
        (1,
         [InputVariableNode(1, 'y', (2, 3), 'float32', 0),
          InputVariableNode(2, 'x', None, None, None)],
         [InputFunctionNode(0, 'Square', [2], [1])],
         ['1 [label="y: (2, 3) float32", color=orange, style=filled]',
          '0 [label="Square", color=lightblue, style=filled, shape=box]', '2 -> 0', '0 -> 1',
          '2 [label="x", color=orange, style=filled]']),

        # same input ids
        (1,
         [InputVariableNode(1, 'y', None, None, 2), InputVariableNode(3, 'x', None, None, None)],
         [InputFunctionNode(2, 'Square', [3, 3], [1])],
         ['1 [label="y", color=orange, style=filled]',
          '2 [label="Square", color=lightblue, style=filled, shape=box]', '3 -> 2', '3 -> 2', '2 -> 1',
          '3 [label="x", color=orange, style=filled]']),

        # complex graph
        (1,
         [InputVariableNode(1, 'y', None, None, 2), InputVariableNode(3, 'x3', None, None, 5),
          InputVariableNode(4, 'x4', None, None, 5), InputVariableNode(6, 'x6', None, None, None),
          InputVariableNode(7, 'x7', None, None, None)],
         [InputFunctionNode(2, 'Square', [3, 4], [1]), InputFunctionNode(5, 'Square2', [6, 7], [3, 4])],
         ['1 [label="y", color=orange, style=filled]',
          '2 [label="Square", color=lightblue, style=filled, shape=box]', '3 -> 2', '4 -> 2', '2 -> 1',
          '3 [label="x3", color=orange, style=filled]',
          '4 [label="x4", color=orange, style=filled]',
          '5 [label="Square2", color=lightblue, style=filled, shape=box]', '6 -> 5', '7 -> 5', '5 -> 3', '5 -> 4',
          '6 [label="x6", color=orange, style=filled]',
          '7 [label="x7", color=orange, style=filled]'
          ]),
    ])
    def test_draw(self, start_variable_id, variable_list, function_list, expected_list):
        graph = Graph(start_variable_id, variable_list, function_list)
        actual = graph.draw(verbose=True)
        expected = 'digraph G {\n' + '\n'.join(expected_list) + '\n}'
        assert actual == expected
