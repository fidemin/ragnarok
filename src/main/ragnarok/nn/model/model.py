from abc import abstractmethod, ABCMeta
from typing import List

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter
from src.main.ragnarok.nn.layer.layer import Layer


class Model(metaclass=ABCMeta):
    params_dict: dict[str, Parameter]
    params = List[Parameter]

    def __init__(self):
        self.params_dict = {}
        self.params = []

    def _append_param(self, name: str, param: Parameter):
        self.params_dict[name] = param
        self.params.append(param)

    @abstractmethod
    def predict(self, *variables: Variable, **kwargs) -> Variable | List[Variable]:
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
                self._append_param(f"{layer.name}__{key}", param)

    def predict(self, *variables: Variable, **kwargs) -> Variable | List[Variable]:
        for layer in self.layers:
            if not type(variables) in (tuple, list):
                variables = [variables]
            variables = layer.forward(*variables, **kwargs)
        return variables
