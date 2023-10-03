import numpy as np

from cnn.layer import Convolution


def test_forward():
    test_input = np.random.rand(10, 3, 4, 4)
    test_W = np.random.rand(5, 3, 3, 3)
    test_b = np.random.rand(5)

    layer = Convolution(test_W, test_b)
    out = layer.forward(test_input)

    assert out.shape == (10, 5, 2, 2)
