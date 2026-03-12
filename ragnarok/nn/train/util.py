from ragnarok.core.tensor import Tensor


def accuracy(y: Tensor, t: Tensor) -> float:
    if y.shape == t.shape:
        t = t.argmax(axis=1)
    y_max_idx = y.argmax(axis=1).reshape(t.shape)
    acc = (y_max_idx == t).mean()
    return acc.data
