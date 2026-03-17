from typing import List

from ragnarok.core.function import Function
from ragnarok.core.tensor import Tensor
from ragnarok.graph.node import DotTensorNode, DotFunctionNode


def draw_tensor(tensor: Tensor, verbose: bool) -> str:
    return DotTensorNode(
        id(tensor),
        name=tensor.name,
        shape=tensor.shape,
        dtype=tensor.dtype,
    ).draw(verbose=verbose)


def draw_function(func: Function) -> str:
    return DotFunctionNode(
        id(func),
        name=func.__class__.__name__,
        input_ids=[id(input_tensor) for input_tensor in func.inputs],
        output_ids=[id(output_weak()) for output_weak in func.outputs],
    ).draw()


class DotGraph:
    def __init__(self, output: Tensor):
        self._output = output

    def draw(self, verbose=False):
        result_list = [draw_tensor(self._output, verbose=verbose)]

        seen_funcs = set()
        drawn_vars = set()
        func_stack: List[Function] = []

        if self._output.creator is not None:
            func_stack.append(self._output.creator)

        while func_stack:
            func = func_stack.pop()
            func_id = id(func)
            if func_id in seen_funcs:
                continue
            seen_funcs.add(func_id)
            result_list.append(draw_function(func))

            for input_tensor in func.inputs:
                input_id = id(input_tensor)
                if input_id not in drawn_vars:
                    result_list.append(draw_tensor(input_tensor, verbose=verbose))
                    drawn_vars.add(input_id)
                if input_tensor.creator is not None:
                    func_stack.append(input_tensor.creator)

        return "digraph G {\n" + "\n".join(result_list) + "\n}"
