import numpy as np

from ragnarok.core.config import Config
from ragnarok.core.function import Function, FunctionVariableError
from ragnarok.core.tensor import Tensor
from ragnarok.core.tensor.dtype import int8


class Dropout(Function):
    _dropout_ratio_key = "dropout_ratio"

    def __init__(self, freeze_mask=False):
        super().__init__()
        self._freeze_mask = freeze_mask

    def forward(self, *variables: Tensor, **kwargs):
        dropout_ratio = kwargs[self._dropout_ratio_key]
        x = variables[0]

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
        keep_ratio = 1 - self.kwargs[self._dropout_ratio_key]
        return dout * self._cache["mask"] / keep_ratio

    def _validate_variables(self, *variables: Tensor, **kwargs):
        if len(variables) != 1:
            raise FunctionVariableError("Dropout requires 1 tensor")

        if self._dropout_ratio_key not in kwargs:
            raise FunctionVariableError("Dropout requires dropout_ratio in kwargs")

        if not 0.0 <= kwargs[self._dropout_ratio_key] <= 1.0:
            raise FunctionVariableError("Dropout ratio should be between 0 and 1")


def dropout(x: Tensor, *, dropout_ratio: float) -> Tensor:
    return Dropout()(x, dropout_ratio=dropout_ratio)
