from abc import ABCMeta, abstractmethod

import numpy as np


class Updater(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: list, grads: list):
        pass


class SGD(Updater):
    def __init__(self, lr=0.01):
        self._lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self._lr * grads[i]

        return params


class Momentum(Updater):
    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._v = None

    def update(self, params: list, grads: list):
        if self._v is None:
            self._v = []
            for param in params:
                self._v.append(np.zeros_like(param))

        for i in range(len(params)):
            self._v[i] = self._momentum * self._v[i] - self._lr * grads[i]
            params[i] += self._v[i]

        return params
