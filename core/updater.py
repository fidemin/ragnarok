from abc import ABCMeta, abstractmethod


class Updater(metaclass=ABCMeta):
    @abstractmethod
    def update(self, params: list, grads: list):
        pass

    @abstractmethod
    def copy(self):
        pass


class SGD(Updater):
    def __init__(self, lr=0.01):
        self._lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self._lr * grads[i]

        return params

    def copy(self):
        return type(self)(lr=self._lr)
