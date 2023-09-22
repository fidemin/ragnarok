from typing import Optional

import numpy as np

from core.layer import Layer, SoftmaxWithLoss
from core.updater import Updater


class Net:
    def __init__(self, layers: list[Layer], updater: Optional[Updater] = None):
        self._layers = layers

        if updater is not None:
            for layer in layers:
                layer.updater = updater.copy()

        self._last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray) -> np.float64:
        y = self.predict(x)
        return self._last_layer.forward(y, t)

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)

        y_max_idx = np.argmax(y, axis=1)
        t_max_idx = np.argmax(t, axis=1)

        return np.sum(y_max_idx == t_max_idx) / float(x.shape[0])

    def gradient_descent(self, x: np.ndarray, t: np.ndarray, dout=1):
        self.loss(x, t)

        dout = self._last_layer.backward(dout)

        for layer in reversed(self._layers):
            dout = layer.backward(dout)

        for layer in reversed(self._layers):
            layer.update_params()

        return dout
