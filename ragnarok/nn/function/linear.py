from ragnarok.core.function import Function, sum_to
from ragnarok.core.function.math import matmul
from ragnarok.core.tensor import Tensor


class Linear(Function):
    def forward(self, *tensors: Tensor, **kwargs):
        x, W, b = tensors

        t = matmul(x, W)
        y = t + b

        t.release()  # data of t is not used anymore
        return y

    def backward(self, *douts: Tensor):
        dout = douts[0]
        x, W, b = self.inputs

        dx = matmul(dout, W.T)
        dW = matmul(x.T, dout)
        db = sum_to(dout, b.shape)

        return dx, dW, db

    def _validate_tensors(self, *tensors: Tensor, **kwargs):
        if len(tensors) != 3:
            raise ValueError("Linear requires 3 tensors")


def linear(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    return Linear()(x, W, b)
