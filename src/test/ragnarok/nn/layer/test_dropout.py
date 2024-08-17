import numpy as np

from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.core.variable.dtype import int32
from src.main.ragnarok.nn.layer.dropout import Dropout


class TestDropout:
    def test_forward(self):
        dropout_ratio = 0.6
        dim1 = 100
        dim2 = 100
        num_of_params = dim1 * dim2
        x = Variable(np.random.rand(dim1, dim2))
        x_avg = np.average(x.data)

        dropout = Dropout(dropout_ratio=dropout_ratio)

        num_of_iter = 1000
        dropped = Variable(0)
        total = 0
        for i in range(num_of_iter):
            y = dropout.forward(x)
            dropped += (y == 0.0).astype(int32).sum()
            total += np.sum(y.data)

        y_avg = Variable(total / num_of_iter / x.size)
        actual_drop_ratio = dropped / num_of_iter / num_of_params

        assert allclose(y_avg, x_avg, atol=0.1)
        assert allclose(actual_drop_ratio, Variable(dropout_ratio), atol=0.1)
