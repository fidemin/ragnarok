import numpy as np

from ragnarok.core.config import Config
from ragnarok.core.function import Function, FunctionTensorError
from ragnarok.core.tensor import Tensor
from ragnarok.core.tensor.dtype import int8


class Dropout(Function):
    _dropout_ratio_key = "dropout_ratio"

    def __init__(self, freeze_mask=False):
        super().__init__()
        self._freeze_mask = freeze_mask

    def forward(self, *tensors: Tensor, **kwargs):
        dropout_ratio = kwargs[self._dropout_ratio_key]
        x = tensors[0]

        if not (self._freeze_mask and "mask" in self._cache):
            self._cache["mask"] = (
                Tensor(np.random.rand(*x.shape)) > dropout_ratio
            ).astype(int8)

        if Config.train:
            keep_ratio = 1 - dropout_ratio
            return x * self._cache["mask"] / keep_ratio
        else:
            return x

    def backward(self, *douts: Tensor):
        dout = douts[0]

        if Config.train:
            keep_ratio = 1 - self.kwargs[self._dropout_ratio_key]
            return dout * self._cache["mask"] / keep_ratio
        else:
            return dout

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 1:
            raise FunctionTensorError("Dropout requires 1 tensor")

        if self._dropout_ratio_key not in kwargs:
            raise FunctionTensorError("Dropout requires dropout_ratio in kwargs")

        if not 0.0 <= kwargs[self._dropout_ratio_key] <= 1.0:
            raise FunctionTensorError("Dropout ratio should be between 0 and 1")


def dropout(x: Tensor, *, dropout_ratio: float) -> Tensor:
    return Dropout()(x, dropout_ratio=dropout_ratio)
