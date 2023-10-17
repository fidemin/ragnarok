import numpy as np

from core.layer import Layer, SoftmaxWithLoss


class Net:
    def __init__(self, layers: list[Layer], use_last_layer=True):
        self._layers = layers
        self._use_last_layer = use_last_layer
        self._last_layer = None

        if self._use_last_layer:
            self._last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray, train_flag=True, kwargs_list=None):
        for i, layer in enumerate(self._layers):
            kwargs = {}
            if kwargs_list is not None:
                kwargs = kwargs_list[i]
            x = layer.forward(x, train_flag=train_flag, **kwargs)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray, kwargs_list=None) -> np.float64:
        y = self.predict(x, kwargs_list=kwargs_list)
        if self._use_last_layer:
            return self._last_layer.forward(y, t)
        else:
            return y

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x, train_flag=False)

        y_max_idx = np.argmax(y, axis=1)
        t_max_idx = np.argmax(t, axis=1)

        return np.sum(y_max_idx == t_max_idx) / float(x.shape[0])

    def gradient_descent(self, x: np.ndarray, t: np.ndarray, dout=1, kwargs_list=None):
        self.loss(x, t, kwargs_list=kwargs_list)

        if self._use_last_layer:
            dout = self._last_layer.backward(dout)

        for layer in reversed(self._layers):
            dout = layer.backward(dout)

        for layer in reversed(self._layers):
            layer.update_params()

        return dout
