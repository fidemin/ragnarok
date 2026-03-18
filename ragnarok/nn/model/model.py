from typing import List, Union

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.module import Module
from ragnarok.nn.layer.activation import get_activation_layer
from ragnarok.nn.layer.linear import Linear


class Sequential(Module):
    _module_list: List[Module]

    def __init__(self, modules: List[Module]):
        super().__init__()
        self._module_list = []

        for i, module in enumerate(modules, start=1):
            if module.name is None:
                layer_name = f"Layer_{i}"
            else:
                layer_name = f"{module.name}_{i}"

            super().__setattr__(layer_name, module)
            self._module_list.append(module)

    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        for module in self._module_list:
            if not type(tensors) in (tuple, list):
                tensors = [tensors]
            tensors = module(*tensors, **kwargs)
        return tensors


class MLP(Module):
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
            layer_name = f"Linear_{i}"
            layer = Linear(out_size=output_size, name=layer_name)
            super().__setattr__(layer_name, layer)
            self._layers.append(layer)

    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        for layer in self._layers[:-1]:
            x = layer(x)
            x = self._activation_layer(x)
        return self._layers[-1](x)
