import numpy as np

from src.main.ragnarok.core.function import Add, Square
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.graph.graph import DotGraph


class TestDotGraph:
    def test_draw_simple(self):
        variable = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        dot_graph = DotGraph(variable)
        result_list = [
            f'{id(variable)} [label="(2, 3) float32", color=orange, style=filled]'
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_one_function(self):
        variable1 = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        variable2 = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        function = Add()
        output = function(variable1, variable2)
        dot_graph = DotGraph(output)

        result_list = [
            f'{id(output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function)} [label="Add", color=lightblue, style=filled, shape=box]',
            f"{id(variable1)} -> {id(function)}",
            f"{id(variable2)} -> {id(function)}",
            f"{id(function)} -> {id(output)}",
            f'{id(variable1)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(variable2)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_one_func_with_same_input(self):
        variable1 = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        function = Add()
        output = function(variable1, variable1)
        dot_graph = DotGraph(output)

        result_list = [
            f'{id(output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function)} [label="Add", color=lightblue, style=filled, shape=box]',
            f"{id(variable1)} -> {id(function)}",
            f"{id(variable1)} -> {id(function)}",
            f"{id(function)} -> {id(output)}",
            f'{id(variable1)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_complex(self):
        variable1 = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        variable2 = Variable(
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        )
        function1 = Add()
        mid_output = function1(variable1, variable2)

        function2 = Square()
        output = function2(mid_output)

        dot_graph = DotGraph(output)

        result_list = [
            f'{id(output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function2)} [label="Square", color=lightblue, style=filled, shape=box]',
            f"{id(mid_output)} -> {id(function2)}",
            f"{id(function2)} -> {id(output)}",
            f'{id(mid_output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function1)} [label="Add", color=lightblue, style=filled, shape=box]',
            f"{id(variable1)} -> {id(function1)}",
            f"{id(variable2)} -> {id(function1)}",
            f"{id(function1)} -> {id(mid_output)}",
            f'{id(variable1)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(variable2)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )
