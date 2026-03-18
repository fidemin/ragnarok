from typing import List

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.module import Module
from ragnarok.nn.core.parameter import Parameter


class TestModule:
    def test_module_params_only(self):
        from ragnarok.nn.core.module import Module

        class Layer(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([0.1, 0.1, 0.1])
                self.bias = Parameter([0.01, 0.02, 0.03])

            def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
                tensor = tensors[0]
                return tensor * self.weight + self.bias

        layer = Layer()
        assert layer._container["weight"] is layer.weight
        assert layer._container["bias"] is layer.bias
        assert layer.params == [layer.weight, layer.bias]

    def test_module_layer_only(self):
        class Layer(Module):
            def __init__(self):
                super().__init__()
                self.weight = Parameter([0.1, 0.1, 0.1])
                self.bias = Parameter([0.01, 0.02, 0.03])

            def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
                tensor = tensors[0]
                return tensor * self.weight + self.bias

        class Model(Module):
            def __init__(self):
                super().__init__()
                self.layer = Layer()

            def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
                tensor = tensors[0]
                return self.layer(tensor)

        model = Model()
        assert model._container["layer"] is model.layer
        assert model.params == [model.layer.weight, model.layer.bias]
