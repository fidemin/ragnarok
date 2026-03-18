from abc import abstractmethod, ABCMeta
from typing import List, Union

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.parameter import Parameter


class Module(metaclass=ABCMeta):
    _container: dict[str, Union[Parameter, "Module"]]
    _name: str

    def __init__(self, name=None):
        self._container = {}

        # TODO: check the name is necessary or not
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __setattr__(self, key, value):
        self._set_init(key, value)
        super().__setattr__(key, value)

    def _set_init(self, key, value):
        # only register Parameter and Module to the container, other attributes are not registered
        if isinstance(value, Parameter) or isinstance(value, Module):
            key_str = f"{key}"
            self._container[key_str] = value

    def __call__(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        return self.forward(*tensors, **kwargs)

    @property
    def params(self) -> List[Parameter]:
        params = []
        for module_or_param in self._container.values():
            if isinstance(module_or_param, Parameter):
                params.append(module_or_param)
            elif isinstance(module_or_param, Module):
                params.extend(module_or_param.params)
        return params

    def zero_grad(self):
        for module_or_param in self._container.values():
            if isinstance(module_or_param, Parameter):
                module_or_param.clear_grad()
            elif isinstance(module_or_param, Module):
                module_or_param.zero_grad()

    @abstractmethod
    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        pass
