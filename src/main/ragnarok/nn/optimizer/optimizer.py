from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional

from src.main.ragnarok.core.variable.variable import zeros_like, Variable
from src.main.ragnarok.nn.core.parameter import Parameter


class Optimizer(metaclass=ABCMeta):
    def update(self, params: Iterable[Parameter]):
        self._pre_update(params)
        for param in params:
            self._update_one(param)

    @abstractmethod
    def _update_one(self, param: Parameter):
        # Implementation required
        pass

    def _pre_update(self, params: Iterable[Parameter]):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def _update_one(self, param: Parameter):
        # NOTE: need to update referenced value
        # NOTE: need to use numpy operation(low level) to avoid creating graph
        param.data -= self.lr * param.grad.data


class Momentum(Optimizer):
    _v: Optional[dict[int, Variable]]

    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._v = None

    def _pre_update(self, params: Iterable[Parameter]):
        if self._v is None:
            self._v = {}
            for param in params:
                self._v[id(param)] = zeros_like(param)

    def _update_one(self, param: Parameter):
        v = self._v[id(param)]
        v.data = self._momentum * v.data - self._lr * param.grad.data
        param.data += v.data
