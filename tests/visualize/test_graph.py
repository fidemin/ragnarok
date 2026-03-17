import numpy as np

from ragnarok.core.function.math import Square, Add
from ragnarok.core.tensor import Tensor
from ragnarok.graph.graph import DotGraph


class TestDotGraph:
    def test_draw_simple(self):
        tensor = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        dot_graph = DotGraph(tensor)
        result_list = [
            f'{id(tensor)} [label="(2, 3) float32", color=orange, style=filled]'
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_one_function(self):
        tensor1 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        tensor2 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        function = Add()
        output = function(tensor1, tensor2)
        dot_graph = DotGraph(output)

        result_list = [
            f'{id(output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function)} [label="Add", color=lightblue, style=filled, shape=box]',
            f"{id(tensor1)} -> {id(function)}",
            f"{id(tensor2)} -> {id(function)}",
            f"{id(function)} -> {id(output)}",
            f'{id(tensor1)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(tensor2)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_one_func_with_same_input(self):
        tensor1 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        function = Add()
        output = function(tensor1, tensor1)
        dot_graph = DotGraph(output)

        result_list = [
            f'{id(output)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(function)} [label="Add", color=lightblue, style=filled, shape=box]',
            f"{id(tensor1)} -> {id(function)}",
            f"{id(tensor1)} -> {id(function)}",
            f"{id(function)} -> {id(output)}",
            f'{id(tensor1)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )

    def test_draw_complex(self):
        tensor1 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        tensor2 = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        function1 = Add()
        mid_output = function1(tensor1, tensor2)

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
            f"{id(tensor1)} -> {id(function1)}",
            f"{id(tensor2)} -> {id(function1)}",
            f"{id(function1)} -> {id(mid_output)}",
            f'{id(tensor1)} [label="(2, 3) float32", color=orange, style=filled]',
            f'{id(tensor2)} [label="(2, 3) float32", color=orange, style=filled]',
        ]
        assert (
            dot_graph.draw(verbose=True)
            == "digraph G {\n" + "\n".join(result_list) + "\n}"
        )
