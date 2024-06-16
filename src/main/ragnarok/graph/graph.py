from collections import namedtuple
from typing import List

from src.main.ragnarok.graph.node import VariableNode, FunctionNode

InputVariableNode = namedtuple('InputVariableNode', ['id', 'name', 'shape', 'dtype', 'create_func_id'])
InputFunctionNode = namedtuple('InputFunctionNode', ['id', 'name', 'input_ids', 'output_ids'])


class Graph:
    def __init__(self, start_variable_id: int, variable_list: List[InputVariableNode], function_list):
        self._start_variable_id = start_variable_id
        self._variable_dict = {variable.id: variable for variable in variable_list}
        self._function_dict = {function.id: function for function in function_list}

    def draw(self, verbose=False):
        result_list = []
        start_input_variable_node = self._variable_dict[self._start_variable_id]
        seen_funcs = set()
        drawn_vars = set()
        func_stack = []
        if start_input_variable_node.create_func_id is not None:
            func_stack.append(start_input_variable_node.create_func_id)
        variable_node = VariableNode(start_input_variable_node.id, name=start_input_variable_node.name,
                                     shape=start_input_variable_node.shape, dtype=start_input_variable_node.dtype)
        result_list.append(variable_node.draw(verbose=verbose))

        while func_stack:
            func_id = func_stack.pop()
            if func_id in seen_funcs:
                continue
            seen_funcs.add(func_id)
            input_func_node = self._function_dict[func_id]
            func_node = FunctionNode(input_func_node.id, name=input_func_node.name, input_ids=input_func_node.input_ids,
                                     output_ids=input_func_node.output_ids)
            result_list.append(func_node.draw())
            for input_id in input_func_node.input_ids:
                input_node = self._variable_dict[input_id]
                if input_id not in drawn_vars:
                    # same variable does not need to be drawn more than once
                    variable_node = VariableNode(input_node.id, name=input_node.name, shape=input_node.shape,
                                                 dtype=input_node.dtype)
                    drawn_vars.add(input_node.id)
                    result_list.append(variable_node.draw(verbose=verbose))
                if input_node.create_func_id is not None:
                    func_stack.append(input_node.create_func_id)

        return 'digraph G {\n' + '\n'.join(result_list) + '\n}'
