from src.main.ragnarok.core.function import Function, matmul, sum_to
from src.main.ragnarok.core.variable import Variable


class Linear(Function):
    def forward(self, *variables: Variable, **kwargs):
        x, W, b = variables

        t = matmul(x, W)
        y = t + b

        t.release()  # data of t is not used anymore
        return y

    def backward(self, *douts: Variable):
        dout = douts[0]
        x, W, b = self.inputs

        dx = matmul(dout, W.T)
        dW = matmul(x.T, dout)
        db = sum_to(dout, b.shape)

        return dx, dW, db

    def _validate_variables(self, *variables: Variable, **kwargs):
        if len(variables) != 3:
            raise ValueError("Linear requires 3 variables")
