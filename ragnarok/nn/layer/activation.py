from typing import List

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.layer import Layer
from ragnarok.nn.function.activation import sigmoid, relu, tanh

CONST_SIGMOID = "sigmoid"
CONST_RELU = "relu"
CONST_TANH = "tanh"


def get_activation_layer(activation_str: str) -> Layer:
    if activation_str == CONST_SIGMOID:
        return Sigmoid()
    elif activation_str == CONST_RELU:
        return ReLU()
    elif activation_str == CONST_TANH:
        return Tanh()
    else:
        raise ValueError(f"Unsupported activation function: {activation_str}")


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
