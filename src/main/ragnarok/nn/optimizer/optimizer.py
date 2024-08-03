from abc import ABCMeta, abstractmethod
from typing import Iterable

from src.main.ragnarok.nn.core.parameter import Parameter


class Optimizer(metaclass=ABCMeta):
    def update(self, params: Iterable[Parameter]):
        for param in params:
            self._update_one(param)

    @abstractmethod
    def _update_one(self, param: Parameter):
        # Implementation required
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def _update_one(self, param: Parameter):
        # NOTE: need to update referenced value
        # NOTE: need to use numpy operation(low level) to avoid creating graph
        param.data -= self.lr * param.grad.data
