import numpy as np
import pytest

from ragnarok.core.function import Square, FunctionVariableError, Exp
from ragnarok.core.util import numerical_diff, allclose
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
        assert allclose(actual, expected)

    @pytest.mark.parametrize('test_input,expected', [
        (Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))),
        (Variable(3), Variable(9)),
        (Variable(3.0), Variable(9.0))
    ])
    def test_forward(self, test_input, expected):
        f = Square()
        actual = f(test_input)
        assert allclose(actual, expected)

    @pytest.mark.parametrize('test_input', [
        [Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))],
        []
    ])
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Square()
            f(*test_input)

    def test_gradient_check(self):
        test_input = Variable(np.array([[1.0, 2.0, 3.0]]))
        f1 = Square()
        f2 = Square()

        f1(test_input)
        actual = f1.backward(Variable(1.0))
        expected = numerical_diff(f2, test_input)
        assert allclose(actual, expected)


class TestExp:
    @pytest.mark.parametrize('test_input,expected', [
        (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.718281828459, 7.389056098931, 20.085536923188]]))),
        (Variable(3), Variable(20.085536923188)),
        (Variable(3.0), Variable(20.085536923188))
    ])
    def test_backward(self, test_input, expected):
        f = Exp()
        f(test_input)
        actual = f.backward(Variable(1.0))
        assert allclose(actual, expected)

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
        assert allclose(actual, expected)

    @pytest.mark.parametrize('test_input', [
        [Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))],
        []
    ])
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Exp()
            f(*test_input)

    def test_gradient_check(self):
        test_input = Variable(np.array([[1.0, 2.0, 3.0]]))
        f1 = Exp()
        f2 = Exp()

        f1(test_input)
        actual = f1.backward(Variable(1.0))
        expected = numerical_diff(f2, test_input)
        assert allclose(actual, expected)


def test_define_by_run():
    test_input = Variable(np.array([0.1, 0.2]))

    f1 = Square()
    f2 = Exp()
    f3 = Square()

    out1 = f1(test_input)
    out2 = f2(out1)
    out3 = f3(out2)

    dout3 = f3.backward(Variable(1.0))
    dout2 = f2.backward(dout3)
    dout1 = f1.backward(dout2)

    assert out3.creator == f3
    assert out3.creator.inputs == (out2,)
    assert out2.creator == f2
    assert out2.creator.inputs == (out1,)
    assert out1.creator == f1
    assert out1.creator.inputs == (test_input,)

    assert dout3 == out2.grad
    assert dout2 == out1.grad
    assert dout1 == test_input.grad
