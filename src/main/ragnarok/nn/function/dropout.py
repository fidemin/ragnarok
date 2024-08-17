import numpy as np

from src.main.ragnarok.core.config import Config
from src.main.ragnarok.core.function import Function, FunctionVariableError
from src.main.ragnarok.core.variable import Variable


class Dropout(Function):
    _dropout_ratio_key = "dropout_ratio"

    def forward(self, *variables: Variable, **kwargs):
        dropout_ratio = kwargs[self._dropout_ratio_key]
        x = variables[0]

        self._cache["mask"] = Variable(
            (np.random.rand(*x.shape) > dropout_ratio).astype(int)
        )

        if Config.train:
            keep_ratio = 1 - dropout_ratio
            return x * self._cache["mask"] / keep_ratio
        else:
            return x

    def backward(self, *douts: Variable):
        dout = douts[0]
        keep_ratio = 1 - self.kwargs[self._dropout_ratio_key]
        return dout * self._cache["mask"] / keep_ratio

    def _validate_variables(self, *variables: Variable, **kwargs):
        if len(variables) != 1:
            raise FunctionVariableError("Dropout requires 1 variable")

        if self._dropout_ratio_key not in kwargs:
            raise FunctionVariableError("Dropout requires dropout_ratio in kwargs")

        if not 0.0 <= kwargs[self._dropout_ratio_key] <= 1.0:
            raise FunctionVariableError("Dropout ratio should be between 0 and 1")
