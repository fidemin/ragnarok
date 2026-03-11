from typing import List

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.nn.function.dropout import dropout
from src.main.ragnarok.nn.layer.layer import Layer


class Dropout(Layer):
    def __init__(self, dropout_ratio: float):
        super().__init__()
        self._dropout_ratio = dropout_ratio

    def _forward(self, *tensors: Tensor, **kwargs) -> List[Tensor]:
        x = tensors[0]
        y = dropout(x, dropout_ratio=self._dropout_ratio)
        return [y]
