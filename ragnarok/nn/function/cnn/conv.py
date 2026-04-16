from ragnarok.core.function import Function, FunctionTensorError
from ragnarok.core.function.math import matmul
from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.cnn.util import _np_img2col, _np_col2img


def img2col(x: Tensor, *, FH: int, FW: int, padding: int, stride: int) -> Tensor:
    col_data = _np_img2col(
        x.data,
        FH=FH,
        FW=FW,
        padding=padding,
        stride=stride,
    )
    return Tensor(col_data)


def col2img(
    dcol_x: Tensor,
    *,
    N: int,
    C: int,
    H: int,
    W: int,
    FH: int,
    FW: int,
    padding: int,
    stride: int
) -> Tensor:
    img_data = _np_col2img(
        dcol_x.data,
        N=N,
        C=C,
        H=H,
        W=W,
        FH=FH,
        FW=FW,
        padding=padding,
        stride=stride,
    )
    return Tensor(img_data)


def fil2col(w_tensor: Tensor) -> Tensor:
    FN = w_tensor.shape[0]

    # returns (FN, FC * FH * FW)
    return w_tensor.reshape(FN, -1)


def col2fil(dcol_W: Tensor, *, FC: int, FH: int, FW: int) -> Tensor:
    # dcol_W.shape: (FN, FC * FH * FW)

    # returns: (FN, FC, FH, FW)
    return dcol_W.reshape((dcol_W.shape[0], FC, FH, FW))


class Conv2D(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        x_var = tensors[0]
        w_var = tensors[1]

        # FC X 1 X 1
        b_var = tensors[2]

        N, C, H, W = x_var.shape
        self._cache["C"] = C
        self._cache["H"] = H
        self._cache["W"] = W

        FN, FC, FH, FW = w_var.shape

        padding = kwargs["padding"]
        stride = kwargs["stride"]

        OH = (H + 2 * padding - FH) // stride + 1
        OW = (W + 2 * padding - FW) // stride + 1

        # col_x.shape: (N * OH * OW, C * FH * FW)
        col_x = img2col(x_var, FH=FH, FW=FW, stride=stride, padding=padding)
        self._cache["col_x"] = col_x

        # col_W.shape: (FN, FC * FH * FW)
        col_W = fil2col(w_var)
        self._cache["col_W"] = col_W

        # out.shape: (N * OH * OW, FN)
        out = matmul(col_x, col_W.T)

        # out.shape: (N, OH, OW, FN)
        out = out.reshape((N, OH, OW, -1))

        # out.shape: (N, FN, OH, OW)
        out = out.transpose(0, 3, 1, 2) + b_var

        return out

    def backward(self, *douts: Tensor):
        # NOTE: higher order gradient is not supported for Conv2D function.
        dout = douts[0]
        N, FN, OH, OW = dout.shape

        # filter shape
        _, FC, FH, FW = self.inputs[1].shape

        # dout_flat.shape: (N, OH, OW, FN) -> (N * OH * OW, FN)
        dout_flat = dout.transpose(0, 2, 3, 1).reshape((N * OH * OW, FN))

        # col_W.shape: (FN, FC * FH * FW)
        col_W = self._cache["col_W"]

        # dcol_x.shape: (N * OH * OW, C * FH * FW)
        dcol_x = matmul(dout_flat, col_W)

        # dx.shape = (N, C, H, W)
        dx = col2img(
            dcol_x,
            N=N,
            C=self._cache["C"],  # C = FC
            H=self._cache["H"],
            W=self._cache["W"],
            FH=FH,
            FW=FW,
            padding=self.kwargs["padding"],
            stride=self.kwargs["stride"],
        )

        # col_x.shape = (N * OH * OW, FC * FH * FW)
        col_x = self._cache["col_x"]

        # dcol_W.shape: (FN, FC * FH * FW)
        dcol_W = matmul(dout_flat.T, col_x)

        dW = col2fil(dcol_W, FC=FC, FH=FH, FW=FW)

        # db.shape: (FN, )
        db = dout_flat.sum(axis=0).reshape((FN, 1, 1))

        return dx, dW, db

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 3:
            raise FunctionTensorError("Conv2D requires 2 tensors")

        if "padding" not in kwargs:
            raise FunctionTensorError("Img2Col requires padding in kwargs")

        if "stride" not in kwargs:
            raise FunctionTensorError("Img2Col requires stride in kwargs")

        C = tensors[0].shape[1]
        FC = tensors[1].shape[1]

        if C != FC:
            raise Exception(
                "Channel of tensor and filter should be same. C: {0}, FC: {1}".format(
                    C, FC
                )
            )
