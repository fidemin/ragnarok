import weakref
from typing import List

import numpy as np

from ragnarok.core.config import Config
from ragnarok.core.tensor import Tensor, to_tensor


class Function:
    inputs: List[Tensor] | None
    outputs: List[Tensor] | None
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
        self, *inputs: int | float | np.ndarray | np.generic | Tensor, **kwargs
    ) -> Tensor | List[Tensor]:
        inputs = [to_tensor(input_) for input_ in inputs]

        self._validate_tensors(*inputs, **kwargs)
        outputs = self.forward(*inputs, **kwargs)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        if Config.enable_backprop:
            # attributes in this part are only used for backpropagation
            this_gen = max([input_.gen for input_ in inputs])
            self.gen = this_gen

            for output in outputs:
                # set strong reference to output tensor
                # self -> output: strong reference
                output.set_creator(self)

            self.inputs = inputs
            # To prevent circular reference, use weakref
            # output -> self: weak reference
            self.outputs = [weakref.ref(output) for output in outputs]
            self.kwargs = kwargs

        return outputs if len(outputs) > 1 else outputs[0]

    def _outputs(self):
        return [output() for output in self.outputs]  # outputs are list of weakref

    def backward(self, *douts: Tensor):
        # NOTE: backward should be implemented based on tensor operation or other forward function
        #   to support high order differentiation
        # DO NOT use numpy operation here
        raise NotImplementedError("Function.backward is not implemented")

    def forward(self, *tensors: Tensor, **kwargs):
        raise NotImplementedError("Function._forward is not implemented")

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        raise NotImplementedError("Function._validate_input is not implemented")


class FunctionTensorError(Exception):
    pass


class NotSupportedOperationException(Exception):
    pass
