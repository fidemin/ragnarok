import math
from abc import abstractmethod, ABCMeta


class LearningRate(metaclass=ABCMeta):
    @abstractmethod
    def learning_rate(self, epoch):
        pass


class InverseSqrt(LearningRate):
    def __init__(self, learning_rate: float):
        self._lr = learning_rate

    def learning_rate(self, epoch: int):
        return self._lr / math.sqrt(float(epoch))
