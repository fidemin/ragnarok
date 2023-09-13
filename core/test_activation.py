import numpy as np

from . import activation


def test_step():
    test_input = np.array([10.0, 0.0, -1.5])

    expected = np.array([1, 0, 0])
    actual = activation.step(test_input)

    assert np.all(actual == expected)

