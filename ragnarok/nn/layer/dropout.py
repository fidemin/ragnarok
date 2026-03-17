from typing import List

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.module import Module
from ragnarok.nn.function.dropout import dropout


class Dropout(Module):
    def __init__(self, dropout_ratio: float):
        super().__init__()
        self._dropout_ratio = dropout_ratio

    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        y = dropout(x, dropout_ratio=self._dropout_ratio)
        return y
