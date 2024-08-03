from typing import List

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import sigmoid, relu, tanh
from src.main.ragnarok.nn.layer.layer import Layer


class Sigmoid(Layer):
    def _forward(self, *variables: Variable, **kwargs) -> List[Variable]:
        x = variables[0]
        y = sigmoid(x)
        return [y]


class Tanh(Layer):
    def _forward(self, *variables: Variable, **kwargs):
        x = variables[0]
        y = tanh(x)
        return [y]


class ReLU(Layer):
    def _forward(self, *variables: Variable, **kwargs) -> List[Variable]:
        x = variables[0]
        y = relu(x)
        return [y]
