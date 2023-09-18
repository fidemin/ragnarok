import numpy as np
import pytest

from core import loss


@pytest.mark.parametrize(
    "y,t,expected",
    [
        (np.array([0.6, 0.1]), np.array([1, 0]), 0.5108256),
        (np.array([[0.6, 0.1], [0.3, 0.5]]), np.array([[1, 0], [0, 1]]), (0.5108256 + 0.6931471) / 2)
    ]
)
def test_cross_entropy(y, t, expected):
    actual = loss.cross_entropy(y, t)
    print(actual)
    assert np.allclose(actual, expected)
