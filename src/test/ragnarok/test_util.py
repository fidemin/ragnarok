import numpy as np
import pytest

from src.main.ragnarok.core.function import Square, Add
from src.main.ragnarok.core.util import numerical_diff, allclose
from src.main.ragnarok.core.variable import Variable


@pytest.mark.parametrize(
    "test_input,expected",
    [(Variable(np.array([3.0, 4.0])), Variable(np.array([6.0, 8.0]))),
     (Variable(np.array([0.0, 2.0])), Variable(np.array([0.0, 4.0])))]
)
def test_numerical_diff__one_input_one_output(test_input, expected):
    f = Square()
    actual = numerical_diff(f, test_input)[0]
    assert np.allclose(expected.data, actual.data)


@pytest.mark.parametrize(
    "test_input,expected",
    [(Variable(np.array([[3.0, 4.0]])), Variable(np.array([[6.0, 8.0]]))),
     (Variable(np.array([[0.0, 2.0]])), Variable(np.array([[0.0, 4.0]])))]
)
def test_numerical_diff__one_input_one_output(test_input, expected):
    f = Square()
    actual = numerical_diff(f, test_input)
    assert np.allclose(expected.data, actual.data)


def test_numerical_diff__two_inputs_one_output():
    f = Add()
    test_inputs = [Variable(np.array([[3.0, 4.0]])), Variable(np.array([[6.0, 8.0]]))]
    expected = Variable(np.array([[1.0, 1.0]])), Variable(np.array([[1.0, 1.0]]))
    actual = numerical_diff(f, *test_inputs)

    for i, dx in enumerate(actual):
        assert allclose(dx, expected[i])
