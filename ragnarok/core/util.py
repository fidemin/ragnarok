from ragnarok.core.function import Function
from ragnarok.core.variable import Variable


def numerical_diff(f: Function, *xs: Variable, eps=1e-4) -> Variable:
    xs_h_minus = [Variable(x.data - eps) for x in xs]
    xs_h_plus = [Variable(x.data + eps) for x in xs]

    ys_h_minus = f(*xs_h_minus)
    ys_h_plus = f(*xs_h_plus)

    return Variable((ys_h_plus.data - ys_h_minus.data) / (2 * eps))
