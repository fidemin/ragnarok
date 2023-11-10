import numpy as np
import pytest

from ragnarok.core.function import Square
from ragnarok.core.util import numerical_diff
from ragnarok.core.variable import Variable


@pytest.mark.parametrize(
    "test_input,expected",
    [(Variable(np.array([3.0, 4.0])), Variable(np.array([6.0, 8.0]))),
     (Variable(np.array([0.0, 2.0])), Variable(np.array([0.0, 4.0])))]
)
def test_numerical_diff(test_input, expected):
    f = Square()
    actual = numerical_diff(f, test_input)
    assert np.allclose(expected.data, actual.data)
