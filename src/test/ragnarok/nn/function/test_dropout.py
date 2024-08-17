import numpy as np

from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, ones_like
from src.main.ragnarok.core.variable.dtype import int32
from src.main.ragnarok.nn.function.dropout import Dropout


class TestDropout:
    def test_forward(self):
        dropout_ratio = 0.6
        dim1 = 100
        dim2 = 100
        num_of_params = dim1 * dim2

        x = Variable(np.random.rand(dim1, dim2))

        x_avg = Variable(np.average(x.data))

        dropout = Dropout()

        num_of_iter = 1000
        dropped = Variable(0)
        total = 0
        for i in range(num_of_iter):
            y = dropout(x, dropout_ratio=dropout_ratio)
            dropped += (y == 0.0).astype(int32).sum()
            total += np.sum(y.data)

        y_avg = Variable(total / num_of_iter / x.size)
        actual_drop_ratio = dropped / num_of_iter / num_of_params

        assert allclose(y_avg, x_avg, atol=0.1)
        assert allclose(actual_drop_ratio, Variable(dropout_ratio), atol=0.1)

    def test_backward(self):
        dropout_ratio = 0.6
        x = Variable(np.random.rand(100, 100))
        dout = ones_like(Variable(np.random.rand(100, 100)))
        dout_avg = np.average(dout.data)

        dropout = Dropout()
        total = 0
        num_of_iter = 1000
        for i in range(num_of_iter):
            y = dropout(x, dropout_ratio=dropout_ratio)
            dx = dropout.backward(dout)

            assert allclose(dx, dout * dropout._cache["mask"] / (1 - dropout_ratio))
            total += np.sum(dx.data)

        y_avg = total / num_of_iter / x.data.size
        assert np.allclose(y_avg, dout_avg, atol=0.1)

    def test_gradient_check(self):
        dropout_ratio = 0.6
        x = Variable(np.random.rand(100, 100))
        dout = ones_like(x)

        dropout = Dropout(freeze_mask=True)

        for_weak_ref = dropout(x, dropout_ratio=dropout_ratio)

        dx = dropout.backward(dout)
        dx_numerical = numerical_diff(dropout, x, dropout_ratio=dropout_ratio)

        assert allclose(dx, dx_numerical)
