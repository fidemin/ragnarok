import numpy as np

from ragnarok.core.function import Function
from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.cnn.conv import img2col
from ragnarok.nn.function.cnn.util import _np_col2img


# class Pooling(Layer):
#     def __init__(self, pool_h: int, pool_w: int, stride: int = 1, padding: int = 0):
#         self._PH = pool_h
#         self._PW = pool_w
#         self._stride = stride
#         self._padding = padding
#
#         self._max_indices = None
#         self._OH = None
#         self._OW = None
#         self._C = None
#         self._W = None
#         self._H = None
#
#     def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
#         N, C, H, W = x.shape
#         self._C = C
#         self._H = H
#         self._W = W
#
#         OH = (H + 2 * self._padding - self._PH) // self._stride + 1
#         OW = (W + 2 * self._padding - self._PW) // self._stride + 1
#         self._OH = OH
#         self._OW = OW
#
#         # col_x.shape: (N * OH * OW, C * PH * PW)
#         col_x = img2col(
#             x, self._PH, self._PW, stride=self._stride, padding=self._padding
#         )
#         # col_x.shape: (N * OH * OW * C, PH * PW)
#         col_x = col_x.reshape(-1, self._PH * self._PW)
#
#         self._max_indices = np.argmax(col_x, axis=1)
#         # out.shape: (N * OH * OW * C, )
#         out = np.max(col_x, axis=1)
#
#         # out.shape = (N, C, OH, OW)
#         out = out.reshape((N, OH, OW, C)).transpose(0, 3, 1, 2)
#
#         return out
#
#     def backward(self, dout: np.ndarray):
#         # dout.shape = (N, C, OH, OW)
#         N = dout.shape[0]
#
#         # dout.shape = (N, OH, OW, C)
#         dout = dout.transpose(0, 2, 3, 1).flatten()
#
#         # dcol_x.shape: (N * OH * OW * C, PH * PW)
#         dcol_x = np.zeros((N * self._OH * self._OW * self._C, self._PH * self._PW))
#         dcol_x[np.arange(self._max_indices.size), self._max_indices] = dout
#
#         # dcol_x.shape: (N * OH * OW, C * PH * PW)
#         dcol_x = dcol_x.reshape(N * self._OH * self._OW, -1)
#         dx = col2img(
#             dcol_x,
#             N,
#             self._C,
#             self._H,
#             self._W,
#             self._PH,
#             self._PW,
#             padding=self._padding,
#             stride=self._stride,
#         )
#
#         # dx.shape: (N, C, H, W)
#         return dx


class MaxPooling(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        x = tensors[0]
        N, C, H, W = x.shape

        self._cache["C"] = C
        self._cache["H"] = H
        self._cache["W"] = W

        PH = kwargs["pool_h"]
        PW = kwargs["pool_w"]

        padding = kwargs["padding"]
        stride = kwargs["stride"]

        OH = (H + 2 * padding - PH) // stride + 1
        OW = (W + 2 * padding - PW) // stride + 1

        self._cache["OH"] = OH
        self._cache["OW"] = OW

        # col_x.shape: (N * OH * OW, C * PH * PW)
        col_x = img2col(x, FH=PH, FW=PW, stride=stride, padding=padding)
        col_x_data = col_x.data

        # col_x.shape: (N * OH * OW * C, PH * PW)
        col_x_data = col_x_data.reshape(-1, PH * PW)

        self._cache["max_indices"] = np.argmax(col_x_data, axis=1)

        # out.shape: (N * OH * OW * C, )
        out_data = np.max(col_x_data, axis=1)

        # out.shape = (N, C, OH, OW)
        out_data = out_data.reshape((N, OH, OW, C)).transpose(0, 3, 1, 2)

        return Tensor(out_data)

    def backward(self, *douts: Tensor):
        # NOTE: higher order gradient is not supported for MaxPooling function.

        dout_data = douts[0].data

        # dout.shape = (N, C, OH, OW)
        N = dout_data.shape[0]

        # dout.shape = (N, OH, OW, C)
        dout_data = dout_data.transpose(0, 2, 3, 1).flatten()

        OH = self._cache["OH"]
        OW = self._cache["OW"]
        C = self._cache["C"]
        PH = self.kwargs["pool_h"]
        PW = self.kwargs["pool_w"]
        max_indices = self._cache["max_indices"]

        H = self._cache["H"]
        W = self._cache["W"]
        padding = self.kwargs["padding"]
        stride = self.kwargs["stride"]

        # dcol_x_data.shape: (N * OH * OW * C, PH * PW)
        dcol_x_data = np.zeros((N * OH * OW * C, PH * PW))
        dcol_x_data[np.arange(max_indices.size), max_indices] = dout_data

        # dcol_x_data.shape: (N * OH * OW, C * PH * PW)
        dcol_x_data = dcol_x_data.reshape(N * OH * OW, -1)

        dx = _np_col2img(
            dcol_x_data,
            N=N,
            C=C,
            H=H,
            W=W,
            FH=PH,
            FW=PW,
            padding=padding,
            stride=stride,
        )

        # dx.shape: (N, C, H, W)
        return dx

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        # TODO: validate should be added
        pass
