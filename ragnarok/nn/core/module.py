from abc import abstractmethod, ABCMeta
from typing import List, Optional

from ragnarok.core.tensor import Tensor
from ragnarok.nn.core.parameter import Parameter


class Module(metaclass=ABCMeta):
    _params: dict[str, Parameter]
    _modules: dict[str, "Module"]
    name: Optional[str]

    def __init__(self, name=None):
        self._params = {}
        self._modules = {}

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    @property
    def params(self) -> dict[str, Parameter]:
        return self._params

    @property
    def modules(self) -> dict[str, "Module"]:
        return self._modules

    def __setattr__(self, key, value):
        self._set_init(key, value)
        super().__setattr__(key, value)

    def _set_init(self, key, value):
        # save parameters in the module's params dictionary with their full names
        if isinstance(value, Parameter):
            param_name = f"{key}"
            self.params[param_name] = value

        if isinstance(value, Module):
            module_name = f"{key}"
            self._modules[module_name] = value

    def __call__(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        return self.forward(*tensors, **kwargs)

    @abstractmethod
    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        pass
