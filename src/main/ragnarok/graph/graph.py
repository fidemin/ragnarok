from typing import List

from src.main.ragnarok.core.function import Function
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.graph.node import DotVariableNode, DotFunctionNode


def draw_variable(variable: Variable, verbose: bool) -> str:
    return DotVariableNode(
        id(variable),
        shape=variable.shape,
        dtype=variable.dtype,
    ).draw(verbose=verbose)


def draw_function(func: Function) -> str:
    return DotFunctionNode(
        id(func),
        name=func.__class__.__name__,
        input_ids=[id(input_variable) for input_variable in func.inputs],
        output_ids=[id(output_weak()) for output_weak in func.outputs],
    ).draw()


class DotGraph:
    def __init__(self, output: Variable):
        self._output = output

    def draw(self, verbose=False):
        result_list = [draw_variable(self._output, verbose=verbose)]

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

            for input_variable in func.inputs:
                input_id = id(input_variable)
                if input_id not in drawn_vars:
                    result_list.append(draw_variable(input_variable, verbose=verbose))
                    drawn_vars.add(input_id)
                if input_variable.creator is not None:
                    func_stack.append(input_variable.creator)

        return "digraph G {\n" + "\n".join(result_list) + "\n}"
