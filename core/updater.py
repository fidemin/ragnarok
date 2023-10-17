from abc import ABCMeta, abstractmethod

import numpy as np


class Updater(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: list[np.ndarray], grads: list[np.ndarray]):
        # This method should update each parameter's all elements in-place.
        # e.g. params[i] += (any operation) or params[i][:] = params[i] + (any operation)"
        pass


class SGD(Updater):
    def __init__(self, lr=0.01):
        self._lr = lr

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        for i in range(len(params)):
            params[i] -= self._lr * grads[i]

        return params


class Momentum(Updater):
    def __init__(self, lr=0.01, momentum=0.9):
        self._lr = lr
        self._momentum = momentum
        self._v = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        if self._v is None:
            self._v = []
            for param in params:
                self._v.append(np.zeros_like(param))

        for i in range(len(params)):
            self._v[i] = self._momentum * self._v[i] - self._lr * grads[i]
            params[i] += self._v[i]

        return params


class AdaGrad(Updater):
    def __init__(self, lr=0.01):
        self._lr = lr
        self._h = None

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        if self._h is None:
            self._h = []
            for param in params:
                self._h.append(np.zeros_like(param))

        for i in range(len(params)):
            self._h[i] += grads[i] * grads[i]
            params[i] -= self._lr / (np.sqrt(self._h[i]) + 1e-7) * grads[i]

        return params


class Adam(Updater):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.99):
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._iter = 0
        self._momentum1 = []
        self._momentum2 = []

    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> list[np.ndarray]:
        if self._iter == 0:
            for param in params:
                self._momentum1.append(np.zeros_like(param))
                self._momentum2.append(np.zeros_like(param))

        self._iter += 1

        for i in range(len(params)):
            self._momentum1[i] = self._beta1 * self._momentum1[i] + (1 - self._beta1) * grads[i]
            momentum1_corr = self._momentum1[i] / (1 - (self._beta1 ** self._iter))
            self._momentum2[i] = self._beta2 * self._momentum2[i] + (1 - self._beta2) * np.power(grads[i], 2)
            momentum2_corr = self._momentum2[i] / (1 - (self._beta2 ** self._iter))

            params[i] -= self._lr * momentum1_corr / (np.sqrt(momentum2_corr + 1e-8))

        return params
