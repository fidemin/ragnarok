from abc import ABCMeta, abstractmethod

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter


class Layer(metaclass=ABCMeta):
    params: dict[str, Parameter]

    def __init__(self):
        self.params = {}

    @abstractmethod
    def forward(self, *variables: Variable, **kwargs):
        pass
