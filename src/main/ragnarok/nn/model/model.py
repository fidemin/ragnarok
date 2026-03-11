from abc import abstractmethod, ABCMeta
from typing import List, Iterable

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.nn.core.layer import Layer
from src.main.ragnarok.nn.core.parameter import Parameter


class Model(metaclass=ABCMeta):
    layers_dict: dict[str, Layer]

    def __init__(self):
        self.layers_dict = {}

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            layer_name = f"{key}"
            # dictionary keeps its insertion order since python 3.7
            self.layers_dict[layer_name] = value

        super().__setattr__(key, value)

    @property
    def params(self) -> Iterable[Parameter]:
        for layer in self.layers_dict.values():
            for param in layer.params.values():
                yield param

    def zero_grad(self):
        for param in self.params:
            param.clear_grad()

    @abstractmethod
    def predict(self, *variables: Tensor, **kwargs) -> Tensor | List[Tensor]:
        pass


class Sequential(Model):
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = []

        for i, layer in enumerate(layers, start=1):
            if layer.name is None:
                layer_name = f"Layer_{i}"
            else:
                layer_name = f"{layer.name}_{i}"

            setattr(self, layer_name, layer)
            self.layers.append(layer)

    def predict(self, *variables: Tensor, **kwargs) -> Tensor | List[Tensor]:
        for layer in self.layers:
            if not type(variables) in (tuple, list):
                variables = [variables]
            variables = layer.forward(*variables, **kwargs)
        return variables
