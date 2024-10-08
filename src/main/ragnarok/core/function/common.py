import weakref
from typing import List

import numpy as np

from src.main.ragnarok.core.config import Config
from src.main.ragnarok.core.variable import Variable, to_variable


class Function:
    inputs: List[Variable] | None
    outputs: List[Variable] | None
    gen: int | None
    _cache: dict
    kwargs: dict

    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.gen = None
        self._cache = {}
        self.kwargs = {}

    def __call__(
        self, *inputs: int | float | np.ndarray | np.generic | Variable, **kwargs
    ) -> Variable | List[Variable]:
        inputs = [to_variable(input_) for input_ in inputs]

        # _validate_variable
        self._validate_variables(*inputs, **kwargs)
        outputs = self.forward(*inputs, **kwargs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        if Config.enable_backprop:
            # these are only used for backpropagation
            this_gen = max([input_.gen for input_ in inputs])
            self.gen = this_gen

            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
            self.kwargs = kwargs

        return outputs if len(outputs) > 1 else outputs[0]

    def _outputs(self):
        return [output() for output in self.outputs]  # outputs are list of weakref

    def backward(self, *douts: Variable):
        # NOTE: backward should be implemented based on variable operation or other forward function
        #   to support high order differentiation
        # DO NOT use numpy operation here
        raise NotImplementedError("Function.backward is not implemented")

    def forward(self, *variables: Variable, **kwargs):
        raise NotImplementedError("Function._forward is not implemented")

    def _validate_variables(self, *variables: Variable, **kwargs):
        raise NotImplementedError("Function._validate_input is not implemented")


class FunctionVariableError(Exception):
    pass


class NotSupportedOperationException(Exception):
    pass
