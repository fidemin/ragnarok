from typing import List

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.nn.function.activation import sigmoid, relu, tanh
from src.main.ragnarok.nn.layer.layer import Layer


class Sigmoid(Layer):
    def _forward(self, *tensors: Tensor, **kwargs) -> List[Tensor]:
        x = tensors[0]
        y = sigmoid(x)
        return [y]


class Tanh(Layer):
    def _forward(self, *tensors: Tensor, **kwargs):
        x = tensors[0]
        y = tanh(x)
        return [y]


class ReLU(Layer):
    def _forward(self, *tensors: Tensor, **kwargs) -> List[Tensor]:
        x = tensors[0]
        y = relu(x)
        return [y]
