import numpy as np
import pytest

from ragnarok.core.function import Square, FunctionVariableError, Exp
from ragnarok.core.variable import Variable


class TestSquare:
    @pytest.mark.parametrize('test_input,expected', [
        (Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[2.0, 4.0, 6.0]]))),
        (Variable(3), Variable(6)),
        (Variable(3.0), Variable(6.0))
    ])
    def test_backward(self, test_input, expected):
        f = Square()
        f(test_input)
        actual = f.backward(Variable(1.0))
        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize('test_input,expected', [
        (Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))),
        (Variable(3), Variable(9)),
        (Variable(3.0), Variable(9.0))
    ])
    def test_forward(self, test_input, expected):
        f = Square()
        actual = f(test_input)
        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize('test_input', [
        [Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))],
        []
    ])
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Square()
            f(*test_input)


class TestExp:
    @pytest.mark.parametrize('test_input,expected', [
        (Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 2.0, 3.0]]))),
        (Variable(3), Variable(3)),
        (Variable(3.0), Variable(3.0))
    ])
    def test_backward(self, test_input, expected):
        f = Exp()
        f(test_input)
        actual = f.backward(Variable(1.0))
        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize('test_input,expected', [
        (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.718281828459, 7.389056098931, 20.085536923188]]))),
        (Variable(3), Variable(20.085536923188)),
        (Variable(3.0), Variable(20.085536923188))
    ])
    def test_forward(self, test_input, expected):
        f = Exp()
        actual = f(test_input)
        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize('test_input', [
        [Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))],
        []
    ])
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Exp()
            f(*test_input)
