from abc import abstractmethod, ABCMeta
from typing import List, Union, Dict

import numpy as np

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

    def __call__(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        return self.forward(*tensors, **kwargs)

    @property
    def params(self) -> List[Parameter]:
        return list(self._params_dict().values())

    @abstractmethod
    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        pass

    def zero_grad(self):
        for module_or_param in self._container.values():
            if isinstance(module_or_param, Parameter):
                module_or_param.clear_grad()
            elif isinstance(module_or_param, Module):
                module_or_param.zero_grad()

    def save(self, path: str):
        params_dict = self._params_dict()

        np_dict = {}
        for param_name, param in params_dict.items():
            np_dict[param_name] = param.data

        np.savez_compressed(path, **np_dict, allow_pickle=False)

    def load(self, path: str):
        params_dict = self._params_dict()

        np_dict = {}
        for param_name, param in params_dict.items():
            np_dict[param_name] = param.data

        np.load(path, **np_dict)

        for param_name, np_array in params_dict.items():
            params_dict[param_name] = np_array

    def _load_one_param(self, param_name: str, param: Parameter):
        param_split = param_name.split("/")

        if len(param_split) == 1:
            self._container[param_split[0]] = param
            return

        # CASE: len(param_split) > 1
        module_name = param_split[0]
        self._container[module_name]._load_one_param("/".join(param_split[1:]), param)

    def _params_dict(self) -> Dict[str, Parameter]:
        params_dict = {}
        for name, module_or_param in self._container.items():
            if isinstance(module_or_param, Parameter):
                params_dict[name] = module_or_param
            elif isinstance(module_or_param, Module):
                for sub_name, sub_param in module_or_param._params_dict().items():
                    params_dict[name + "/" + sub_name] = sub_param
            else:
                raise ValueError(
                    f"{type(module_or_param)} is not supported type for module container"
                )
        return params_dict

    def _set_init(self, key, value):
        # only register Parameter and Module to the container, other attributes are not registered
        if isinstance(value, Parameter) or isinstance(value, Module):
            key_str = f"{key}"
            self._container[key_str] = value
