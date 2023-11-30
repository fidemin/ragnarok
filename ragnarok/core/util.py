import numpy as np

from ragnarok.core.function import Function
from ragnarok.core.variable import Variable


def numerical_diff(f: Function, *xs: Variable, eps=1e-4):
    dxs = [Variable(np.zeros_like(x.data)) for x in xs]

    for i, x in enumerate(xs):
        data = x.data
        index_iter = np.ndindex(data.shape)

        try:
            while True:
                idx = next(index_iter)
                original_value = data[idx]

                data[idx] = original_value - eps
                ys_h_minus = f(*xs)

                data[idx] = original_value + eps
                ys_h_plus = f(*xs)

                data[idx] = original_value

                dys = (ys_h_plus.data - ys_h_minus.data) / (2 * eps)
                dy = np.sum(dys)
                dxs[i].data[idx] = dy

        except StopIteration:
            pass

    return dxs if len(dxs) > 1 else dxs[0]


def allclose(var1: Variable, var2: Variable, atol=1.e-8):
    return np.allclose(var1.data, var2.data, atol=atol)
