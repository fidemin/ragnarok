from abc import abstractmethod, ABCMeta
from typing import List

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter
from src.main.ragnarok.nn.layer.layer import Layer


class Model(metaclass=ABCMeta):
    params: dict[str, Parameter]

    def __init__(self):
        self.params = {}

    @abstractmethod
    def forward(self, *variables: Variable, **kwargs):
        pass


class Sequential(Model):
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = []

        for i, layer in enumerate(layers, start=1):
            self.layers.append(layer)

            if layer.name is None:
                layer.name = f"Layer__{i}"
            else:
                layer.name = f"{layer.name}__{i}"

            for key, param in layer.params.items():
                self.params[f"{layer.name}__{key}"] = param

    def forward(self, *variables: Variable, **kwargs):
        for layer in self.layers:
            variables = layer.forward(*variables, **kwargs)
        return variables
