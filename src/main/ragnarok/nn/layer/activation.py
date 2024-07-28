from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import sigmoid
from src.main.ragnarok.nn.layer.layer import Layer


class Sigmoid(Layer):
    def forward(self, *variables: Variable, **kwargs):
        x = variables[0]
        y = sigmoid(x)
        return y
