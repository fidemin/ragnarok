import numpy as np

from . import activation


def test_step():
    test_input = np.array([10.0, 0.0, -1.5])

    expected = np.array([1, 0, 0])
    actual = activation.step(test_input)

    assert np.all(actual == expected)


def test_sigmoid():
    test_input = np.array([10.0, 0.0, -1.5])

    expected = np.array([0.9999546021312978, 0.5, 0.18242552380627775])
    actual = activation.sigmoid(test_input)

    assert np.allclose(actual, expected)


def test_softmax():
    test_input = np.array([0.3, 1.5, 1])

    expected = np.array([0.15788136769202, 0.52418460065905, 0.31793403164894])
    actual = activation.softmax(test_input)

    assert np.allclose(actual, expected)
