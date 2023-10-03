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
        N, C, height, width = x.shape
        fil_N, fil_C, fil_h, fil_w = self._W.shape

        out_h = (height + 2 * self._padding - fil_h) // self._stride + 1
        out_w = (width + 2 * self._padding - fil_w) // self._stride + 1

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


class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, padding: int = 0):
        self._PH = pool_h
        self._PW = pool_w
        self._stride = stride
        self._padding = padding

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        N, C, H, W = x.shape

        OH = (H + 2 * self._padding - self._PH) // self._stride + 1
        OW = (W + 2 * self._padding - self._PW) // self._stride + 1

        col_x = img2col(x, self._PH, self._PW, stride=self._stride, padding=self._padding)
        col_x = col_x.reshape(-1, self._PH * self._PW)

        out = np.max(col_x, axis=1)

        # out.shape = (N, C, OH, OW)
        out = out.reshape((N, OH, OW, C)).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout: np.ndarray):
        # TODO: need to be implemented
        pass

    def update_params(self):
        # There is no parameters for pooling
        pass
