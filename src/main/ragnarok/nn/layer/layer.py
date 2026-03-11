from abc import ABCMeta, abstractmethod
from typing import Optional, List

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.nn.core.parameter import Parameter


class Layer(metaclass=ABCMeta):
    params: dict[str, Parameter]
    name: Optional[str]

    def __init__(self, name=None):
        self.params = {}
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def forward(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        out_vars = self._forward(*tensors, **kwargs)
        return out_vars[0] if len(out_vars) == 1 else out_vars

    @abstractmethod
    def _forward(self, *tensors: Tensor, **kwargs) -> List[Tensor]:
        pass
