import numpy as np

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.layer.dropout import Dropout


class TestDropout:
    def test_forward(self):
        dropout_ratio = 0.6
        x = Variable(np.random.rand(100, 100))
        x_avg = np.average(x.data)

        dropout = Dropout(dropout_ratio=dropout_ratio)

        num_of_iter = 1000
        total = 0
        for i in range(num_of_iter):
            y = dropout.forward(x)
            total += np.sum(y.data)

        y_avg = total / num_of_iter / x.data.size

        assert np.allclose(y_avg, x_avg, atol=0.1)
