from abc import abstractmethod, ABCMeta
from typing import List, Iterable, Union

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.layer import Layer
from ragnarok.nn.core.module import Module
from ragnarok.nn.core.parameter import Parameter
from ragnarok.nn.layer.activation import get_activation_layer
from ragnarok.nn.layer.linear import Linear


class Model(metaclass=ABCMeta):
    layers_dict: dict[str, Layer]

    def __init__(self):
        self.layers_dict = {}

    def __setattr__(self, key, value):
        if isinstance(value, Layer) or isinstance(value, Module):
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
    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        pass


class Sequential(Model):
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = []

        for i, layer in enumerate(layers, start=1):
            if layer.name is None:
                layer_name = f"Layer_{i}"
            else:
                layer_name = f"{layer.name}_{i}"

            super().__setattr__(layer_name, layer)
            self.layers.append(layer)

    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        for layer in self.layers:
            if not type(tensors) in (tuple, list):
                tensors = [tensors]
            tensors = layer.forward(*tensors, **kwargs)
        return tensors


class MLP(Model):
    def __init__(
        self,
        *,
        out_sizes: Union[list[int], tuple[int, ...]],
        activation: str = "relu",
    ):
        super().__init__()
        self._fc_output_sizes = out_sizes
        self._activation_layer = get_activation_layer(activation)
        self._layers = []

        for i, output_size in enumerate(out_sizes, start=1):
            layer_name = f"linear_{i}"
            layer = Linear(out_size=output_size, name=layer_name)
            super().__setattr__(layer_name, layer)
            self._layers.append(layer)

    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        for layer in self._layers[:-1]:
            x = layer(x)
            x = self._activation_layer(x)
        return self._layers[-1](x)
