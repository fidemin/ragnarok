import os

import pytest

from src.main.ragnarok.graph.graph import Graph, InputVariableNode, InputFunctionNode
from src.main.ragnarok.graph.plot import plot_graph


@pytest.mark.parametrize('start_variable_id, variable_list, function_list, output_file', [
    (0,
     [InputVariableNode(0, 'x', (2, 3), 'float32', 1),
      InputVariableNode(1, 'Square', None, None, None)],
     [InputFunctionNode(1, 'Square', [0], [1])],
     'temp/test_output.png')
])
def test_plot_graph(start_variable_id, variable_list, function_list, output_file):
    graph = Graph(start_variable_id, variable_list, function_list)
    plot_graph(graph, output_file=output_file, temp_dir='temp')
    assert os.path.isfile(output_file)
