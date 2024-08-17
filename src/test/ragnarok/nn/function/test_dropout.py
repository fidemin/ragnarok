import numpy as np

from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable, ones_like
from src.main.ragnarok.nn.function.dropout import Dropout


class TestDropout:
    def test_forward(self):
        dropout_ratio = 0.6
        x = Variable(np.random.rand(100, 100))
        x_sum = np.average(x.data)

        dropout = Dropout()

        num_of_iter = 1000
        total = 0
        for i in range(num_of_iter):
            y = dropout(x, dropout_ratio=dropout_ratio)
            total += np.sum(y.data)

        y_sum = total / num_of_iter / x.data.size

        assert np.allclose(y_sum, x_sum, atol=0.1)

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
