from abc import ABCMeta, abstractmethod
from typing import Iterable, Optional, List

import numpy as np

from src.main.ragnarok.core.variable.variable import zeros_like, Variable
from src.main.ragnarok.nn.core.parameter import Parameter


class Optimizer(metaclass=ABCMeta):
    def update(self, params: Iterable[Parameter]):
        self._pre_update(params)
        for idx, param in enumerate(params):
            self._update_one(idx, param)

    @abstractmethod
    def _update_one(self, idx: int, param: Parameter):
        # Implementation required
        pass

    def _pre_update(self, params: Iterable[Parameter]):
        pass


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def _update_one(self, idx: int, param: Parameter):
        # NOTE: need to update referenced value
        # NOTE: need to use numpy operation(low level) to avoid creating graph
        param.data -= self.lr * param.grad.data


class Momentum(Optimizer):
    _v: Optional[List[Variable]]

    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._v = None

    def _pre_update(self, params: Iterable[Parameter]):
        if self._v is None:
            self._v = []
            for param in params:
                self._v.append(zeros_like(param))

    def _update_one(self, idx: int, param: Parameter):
        v = self._v[idx]
        v.data = self._momentum * v.data - self._lr * param.grad.data
        param.data += v.data


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = 0
        self._momentum1 = []
        self._momentum2 = []
        self._epsilon = 1e-8

    def _pre_update(self, params: Iterable[Parameter]):
        if self._iter == 0:
            for param in params:
                self._momentum1.append(zeros_like(param))
                self._momentum2.append(zeros_like(param))

        self._iter += 1

    def _update_one(self, idx: int, param: Parameter):
        self._momentum1[idx].data = (
            self._beta1 * self._momentum1[idx].data
            + (1 - self._beta1) * param.grad.data
        )
        momentum1_corr = self._momentum1[idx].data / (1 - (self._beta1**self._iter))

        self._momentum2[idx].data = self._beta2 * self._momentum2[idx].data + (
            1 - self._beta2
        ) * np.power(param.grad.data, 2)
        momentum2_corr = self._momentum2[idx].data / (1 - (self._beta2**self._iter))

        param.data -= (
            self._lr * momentum1_corr / (np.sqrt(momentum2_corr) + self._epsilon)
        )
