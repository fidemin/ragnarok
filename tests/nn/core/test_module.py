from typing import List

import pytest

from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose
from ragnarok.nn.core.module import Module
from ragnarok.nn.core.parameter import Parameter


@pytest.fixture
def param_only_module_class():
    class Layer(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter([0.1, 0.1, 0.1])
            self.bias = Parameter([0.01, 0.02, 0.03])

        def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
            tensor = tensors[0]
            return tensor * self.weight + self.bias

    return Layer


@pytest.fixture
def layer_only_module_class():
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

    return Model


@pytest.fixture
def complex_module_class():
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
            self.weight2 = Parameter([0.1])
            self.bias2 = Parameter([0.0])

        def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
            tensor = tensors[0]
            y = self.layer(tensor)
            y = y * self.weight2 + self.bias2
            return y

    return Model


class TestModule:
    def test_init_params_only(self, param_only_module_class):
        module = param_only_module_class()
        assert module._container["weight"] is module.weight
        assert module._container["bias"] is module.bias
        assert module.params == [
            module.weight,
            module.bias,
        ]
        assert module._params_dict() == {
            "weight": module.weight,
            "bias": module.bias,
        }

    def test_init_layer_only(self, layer_only_module_class):
        module = layer_only_module_class()
        assert module._container["layer"] is module.layer
        assert module.params == [
            module.layer.weight,
            module.layer.bias,
        ]
        assert module._params_dict() == {
            "layer/weight": module.layer.weight,
            "layer/bias": module.layer.bias,
        }

    def test_init_with_layer_and_param(self, complex_module_class):
        module = complex_module_class()
        assert module._container["layer"] is module.layer
        assert module._container["weight2"] is module.weight2
        assert module._container["bias2"] is module.bias2
        assert module.params == [
            module.layer.weight,
            module.layer.bias,
            module.weight2,
            module.bias2,
        ]
        assert module._params_dict() == {
            "layer/weight": module.layer.weight,
            "layer/bias": module.layer.bias,
            "weight2": module.weight2,
            "bias2": module.bias2,
        }

    def test_save_and_load(self):
        class Layer(Module):
            def __init__(self, multiplier: float):
                super().__init__()
                self.weight = multiplier * Parameter([0.1, 0.1, 0.1])
                self.bias = multiplier * Parameter([0.01, 0.02, 0.03])

            def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
                tensor = tensors[0]
                return tensor * self.weight + self.bias

        class Model(Module):
            def __init__(self, layer: Layer, multiplier: float):
                super().__init__()
                self.layer = layer
                self.weight2 = Parameter([0.1]) * multiplier
                self.bias2 = Parameter([0.2]) * multiplier

            def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
                tensor = tensors[0]
                y = self.layer(tensor)
                y = y * self.weight2 + self.bias2
                return y

        layer1 = Layer(2.0)
        module_for_save = Model(layer1, 2.0)

        tensor = Tensor([[0.1, 0.2, 0.3]])
        expected = module_for_save(tensor)

        file_name = "temp_test_module_1.npz"
        module_for_save.save(file_name)

        layer2 = Layer(1.5)
        module_for_load = Model(layer2, 1.5)

        actual = module_for_load(tensor)

        # before load, result should be different
        assert not allclose(expected, actual)

        # after load, result should be same
        module_for_load.load(file_name)
        actual = module_for_load(tensor)
        assert not allclose(expected, actual)
