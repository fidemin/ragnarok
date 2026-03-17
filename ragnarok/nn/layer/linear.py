from typing import List

import numpy as np

from ragnarok.core.tensor import Tensor
from ragnarok.core.tensor.dtype import float32
from ragnarok.core.tensor.tensor import zeros
from ragnarok.nn.core.module import Module
from ragnarok.nn.core.parameter import Parameter
from ragnarok.nn.function.linear import linear


class Linear(Module):
    def __init__(
        self,
        out_size: int,
        in_size: int = None,
        *,
        use_bias: bool = True,
        dtype=float32,
        name=None,
    ):
        super().__init__(name=name)

        self.out_size = out_size
        self.in_size = in_size
        self.use_bias = use_bias
        self.dtype = dtype

        if self.in_size is not None:  # for lazy initialization
            self._init_params()

    def _init_params(self):
        init_weight = 0.01
        data_W = init_weight * np.random.randn(self.in_size, self.out_size).astype(
            dtype=self.dtype
        )
        param_W = Parameter(data_W, name="W")
        self.W = param_W

        if self.use_bias:
            data_b = np.zeros(self.out_size).astype(dtype=self.dtype)
            param_b = Parameter(data_b, name="b")
            self.b = param_b

    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        if self.in_size is None:
            self.in_size = x.shape[-1]  # last dimension of x
            self._init_params()

        W = self.W

        if self.use_bias:
            b = self.b
        else:
            b = zeros(self.out_size, dtype=self.dtype)

        out_var = linear(x, W, b)
        return out_var
