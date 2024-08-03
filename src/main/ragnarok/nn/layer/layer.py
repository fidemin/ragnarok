from abc import ABCMeta, abstractmethod
from typing import Optional, List

from src.main.ragnarok.core.variable import Variable
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

    def forward(self, *variables: Variable, **kwargs) -> Variable | List[Variable]:
        out_vars = self._forward(*variables, **kwargs)
        return out_vars[0] if len(out_vars) == 1 else out_vars

    @abstractmethod
    def _forward(self, *variables: Variable, **kwargs) -> List[Variable]:
        pass
