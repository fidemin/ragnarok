import numpy as np

from src.main.ragnarok.core.function import Function
from src.main.ragnarok.core.variable import Variable


class Conv2D(Function):

    def backward(self, *douts: Variable):
        pass

    def forward(self, *variables: Variable, **kwargs):
        pass

    def _validate_variables(self, *variables: Variable, **kwargs):
        pass


class Img2Col(Function):
    def forward(self, *variables: Variable, **kwargs):
        x_var = variables[0]
        x_data = x_var.data

        N, C, H, W = x_data.shape
        self._cache["N"] = N
        self._cache["C"] = C
        self._cache["H"] = H
        self._cache["W"] = W

        FH = kwargs["FH"]
        FW = kwargs["FW"]
        padding = kwargs["padding"]
        stride = kwargs["stride"]

        OH = (H + 2 * padding - FH) // stride + 1
        OW = (W + 2 * padding - FW) // stride + 1

        img = np.pad(x_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        col_data = np.zeros((N, C, OH, OW, FH, FW))

        for out_y in range(OH):
            y = out_y * stride
            y_max = y + FH

            for out_x in range(OW):
                x_data = out_x * stride
                x_max = x_data + FW
                col_data[:, :, out_y, out_x, :, :] = img[:, :, y:y_max, x_data:x_max]

        # col.shape: (N, C, OH, OW, FH, FW) -> (N, OH, OW, C, FH, FW)
        col_data = col_data.transpose((0, 2, 3, 1, 4, 5))
        # col.shape: (N, OH, OW, C, FH, FW) -> (N * OH * OW, C * FH * FW)
        col_data = col_data.reshape((N * OH * OW, -1))
        return Variable(col_data)

    def backward(self, *douts: Variable):
        # NOTE: This function uses numpy for implementation. High order differentiation is not supported.
        dout_var = douts[0]
        dout_data = dout_var.data

        N = self._cache["N"]
        C = self._cache["C"]
        H = self._cache["H"]
        W = self._cache["W"]
        FH = self.kwargs["FH"]
        FW = self.kwargs["FW"]
        padding = self.kwargs["padding"]
        stride = self.kwargs["stride"]

        OH = (H + 2 * padding - FH) // stride + 1
        OW = (W + 2 * padding - FW) // stride + 1

        col = dout_data.reshape((N, OH, OW, C, FH, FW))
        # col.shape: (N, C, OH, OW, FH, FW)
        col = col.transpose(0, 3, 1, 2, 4, 5)
        img = np.pad(
            np.zeros((N, C, H, W)),
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        )

        for out_y in range(OH):
            y = out_y * stride
            y_max = y + FH

            for out_x in range(OW):
                x = out_x * stride
                x_max = x + FW
                img[:, :, y:y_max, x:x_max] += col[:, :, out_y, out_x, :, :]

        # target shape: (N, C, H, W)
        img_data = img[:, :, padding : H + padding, padding : W + padding]
        return Variable(img_data)

    def _validate_variables(self, *variables: Variable, **kwargs):
        if len(variables) != 1:
            raise ValueError("Img2Col requires 1 variable")

        if "FH" not in kwargs:
            raise ValueError("Img2Col requires FH in kwargs")

        if "FW" not in kwargs:
            raise ValueError("Img2Col requires FW in kwargs")

        if "padding" not in kwargs:
            raise ValueError("Img2Col requires padding in kwargs")

        if "stride" not in kwargs:
            raise ValueError("Img2Col requires stride in kwargs")
