import numpy as np

from cnn.util import img2col, fil2col, col2fil, col2img
from core.layer import Layer
from core.updater import Updater


class Convolution(Layer):
    def __init__(self, F: np.ndarray, b: np.ndarray, updater: Updater, stride=1, padding=0):
        # self._W.shape: (FN, FC, FH, FW)
        self._F = F
        self._dF = None

        # self._b.shape: (FN, 1)
        self._b = b
        self._db = None

        self._updater = updater

        self._stride = stride
        self._padding = padding

        self._col_W = None
        self._col_x = None

        self._C = None
        self._H = None
        self._W = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        N, C, H, W = x.shape
        self._C = C
        self._H = H
        self._W = W

        FN, FC, FH, FW = self._F.shape

        # C = FC
        if C != FC:
            # TODO: Exception should be more specific
            raise Exception("Channel of X and filter should be same. C: {0}, FC: {1}".format(C, FC))

        OH = (H + 2 * self._padding - FH) // self._stride + 1
        OW = (W + 2 * self._padding - FW) // self._stride + 1

        # col_x.shape: (N * OH * OW, C * FH * FW)
        col_x = img2col(x, FH, FW, stride=self._stride, padding=self._padding)
        self._col_x = col_x

        # col_W.shape: (FN, FC * FH * FW)
        col_W = fil2col(self._F)
        self._col_W = col_W

        # out.shape: (N * OH * OW, FN)
        out = np.matmul(col_x, col_W.T) + self._b

        # out.shape: (N, OH, OW, FN)
        out = out.reshape((N, OH, OW, -1))

        # out.shape: (N, FN, OH, OW)
        out = out.transpose(0, 3, 1, 2)

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, FN, OH, OW = dout.shape
        _, FC, FH, FW = self._F.shape

        # dout.shape: (N, OH, OW, FN)
        dout = dout.transpose(0, 2, 3, 1)

        # dout.shape: (N * OH * OW, FN)
        dout = dout.reshape((N * OH * OW, FN))

        # dcol_x.shape: (N * OH * OW, C * FH * FW)
        # self._col_W.shape: (FN, FC * FH * FW)
        dcol_x = np.matmul(dout, self._col_W)
        dx = col2img(dcol_x, N, self._C, self._H, self._W, FH, FW, padding=self._padding, stride=self._stride)

        # dcol_F.shape: (FN, FC * FH * FW)
        # col_x.shape: (N * OH * OW, C * FH * FW)
        dcol_F = np.matmul(dout.T, self._col_x)

        self._dF = col2fil(dcol_F, FC, FH, FW)

        # db.shape: (FN, )
        self._db = np.sum(dout, axis=0).flatten()

        return dx

    def update_params(self):
        params = self._updater.update([self._F, self._b], [self._dF, self._db])
        self._F = params[0]
        self._b = params[1]


class Pooling(Layer):
    def __init__(self, pool_h: int, pool_w: int, stride: int = 1, padding: int = 0):
        self._PH = pool_h
        self._PW = pool_w
        self._stride = stride
        self._padding = padding

        self._max_indices = None
        self._OH = None
        self._OW = None
        self._C = None
        self._W = None
        self._H = None

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        N, C, H, W = x.shape
        self._C = C
        self._H = H
        self._W = W

        OH = (H + 2 * self._padding - self._PH) // self._stride + 1
        OW = (W + 2 * self._padding - self._PW) // self._stride + 1
        self._OH = OH
        self._OW = OW

        # col_x.shape: (N * OH * OW, C * PH * PW)
        col_x = img2col(x, self._PH, self._PW, stride=self._stride, padding=self._padding)
        # col_x.shape: (N * OH * OW * C, PH * PW)
        col_x = col_x.reshape(-1, self._PH * self._PW)

        self._max_indices = np.argmax(col_x, axis=1)
        # out.shape: (N * OH * OW * C, )
        out = np.max(col_x, axis=1)

        # out.shape = (N, C, OH, OW)
        out = out.reshape((N, OH, OW, C)).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout: np.ndarray):
        # dout.shape = (N, C, OH, OW)
        N = dout.shape[0]

        # dout.shape = (N, OH, OW, C)
        dout = dout.transpose(0, 2, 3, 1).flatten()

        # dcol_x.shape: (N * OH * OW * C, PH * PW)
        dcol_x = np.zeros((N * self._OH * self._OW * self._C, self._PH * self._PW))
        dcol_x[np.arange(self._max_indices.size), self._max_indices] = dout

        # dcol_x.shape: (N * OH * OW, C * PH * PW)
        dcol_x = dcol_x.reshape(N * self._OH * self._OW, -1)
        dx = col2img(dcol_x, N, self._C, self._H, self._W, self._PH, self._PW, padding=self._padding,
                     stride=self._stride)

        # dx.shape: (N, C, H, W)
        return dx

    def update_params(self):
        # There is no parameters for pooling
        pass
