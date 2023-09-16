import numpy as np
import pytest

from . import numerical


def func1(x: np.ndarray) -> np.ndarray:
    return np.power(x, 2)


@pytest.mark.parametrize(
    "test_input,expected",
    [(np.array([3.0, 4.0]), np.array([6.0, 8.0])), (np.array([0.0, 2.0]), np.array([0.0, 4.0]))]
)
def test_gradient(test_input, expected):
    actual = numerical.gradient(func1, test_input)
    assert np.allclose(expected, actual)
