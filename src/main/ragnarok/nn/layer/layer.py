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

    def forward(self, *variables: Variable, **kwargs) -> List[Variable]:
        out = self._forward(*variables, **kwargs)
        return self._to_variable_list(out)

    @abstractmethod
    def _forward(self, *variables: Variable, **kwargs) -> Variable | List[Variable]:
        pass

    def _to_variable_list(self, x: Variable | List[Variable]):
        if not isinstance(x, list):
            x = [x]
        return x
