import numpy as np

from cnn.util import img2col, fil2col
from core.layer import Layer


class Convolution(Layer):
    def __init__(self, W: np.ndarray, b: np.ndarray, stride=1, padding=0):
        self._W = W
        self._b = b
        self._stride = stride
        self._padding = padding

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        N, C, width, height = x.shape
        fil_N, fil_C, fil_h, fil_w = self._W.shape

        out_h = (width + 2 * self._padding - fil_h) // self._stride + 1
        out_w = (height + 2 * self._padding - fil_w) // self._stride + 1

        col_x = img2col(x, fil_h, fil_w, stride=self._stride, padding=self._padding)
        col_W = fil2col(self._W)
        out = np.matmul(col_x, col_W.T) + self._b
        out = out.reshape((N, out_h, out_w, -1))
        out = out.transpose(0, 3, 1, 2)

        return out

    def backward(self, dout: np.ndarray):
        # TODO: need to be implemented
        pass

    def update_params(self):
        # TODO: need to be implemented
        pass
