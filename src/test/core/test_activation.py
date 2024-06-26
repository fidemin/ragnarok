import numpy as np
import pytest

from src.main.core import activation


def test_step():
    test_input = np.array([10.0, 0.0, -1.5])

    expected = np.array([1, 0, 0])
    actual = activation.step(test_input)

    assert np.all(actual == expected)


def test_sigmoid():
    test_input = np.array([10.0, 0.0, -1.5, -800])

    expected = np.array([0.9999546021312978, 0.5, 0.18242552380627775, 0.000])
    actual = activation.sigmoid(test_input)

    assert np.allclose(actual, expected)


def test_tanh():
    test_input = np.array([10.0, 0.0, -1.5, -800, 800])

    expected = np.array([0.999999995878, 0.0, -0.905148253645, -1.0, 1.0])
    actual = activation.tanh(test_input)

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # test case 1
        (np.array([[0.3, 1.5, 1], [1.0, 2.0, 2.0]]),
         np.array([[0.15788136769202, 0.52418460065905, 0.31793403164894], [0.155362, 0.422319, 0.422319]])),

        # test case 2
        (np.array([0.3, 1.5, 1]),
         np.array([0.15788136769202, 0.52418460065905, 0.31793403164894]))]
)
def test_softmax(test_input, expected):
    actual = activation.softmax(test_input)

    assert np.allclose(actual, expected)
