import numpy as np
import pytest

from ragnarok.core.function import Square, FunctionVariableError, Exp, Add
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
        assert allclose(actual[0], expected)

    @pytest.mark.parametrize('test_input,expected', [
        (Variable(np.array([[1.0, 2.0, 3.0]])), Variable(np.array([[1.0, 4.0, 9.0]]))),
        (Variable(3), Variable(9)),
        (Variable(3.0), Variable(9.0))
    ])
    def test_call(self, test_input, expected):
        f = Square()
        actual = f(test_input)
        assert actual.creator is f
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
        assert allclose(actual[0], expected)


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
        assert allclose(actual[0], expected)

    @pytest.mark.parametrize('test_input,expected', [
        (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.718281828459, 7.389056098931, 20.085536923188]]))),
        (Variable(3), Variable(20.085536923188)),
        (Variable(3.0), Variable(20.085536923188))
    ])
    def test_call(self, test_input, expected):
        f = Exp()
        actual = f(test_input)
        assert actual.creator is f
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
        assert allclose(actual[0], expected)


class TestAdd:
    def test_forward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Add()

        expected = Variable(np.array([0.4, 0.6]))
        actual = f.forward(test_input1, test_input2)

        assert np.allclose(actual.data, expected.data)

    def test_backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Add()
        f(test_input1, test_input2)
        dout = Variable(np.array([1.0, 2.0]))

        expected0 = Variable(np.array([1.0, 2.0]))
        expected1 = Variable(np.array([1.0, 2.0]))

        dx0, dx1 = f.backward(dout)

        assert np.allclose(dx0.data, expected0.data)
        assert np.allclose(dx1.data, expected1.data)

    def test_gradient_check(self):
        f = Add()
        test_inputs = [Variable(np.array([[3.0, 4.0]])), Variable(np.array([[6.0, 8.0]]))]

        f(*test_inputs)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


def test_define_by_run():
    test_input = Variable(np.array([0.1, 0.2]))

    f1 = Square()
    f2 = Exp()
    f3 = Square()

    out1 = f1(test_input)
    out2 = f2(out1)
    out3 = f3(out2)

    dout3 = f3.backward(Variable(1.0))
    dout2 = f2.backward(dout3[0])
    f1.backward(dout2[0])

    assert out3.creator == f3
    assert out3.creator.inputs == (out2,)
    assert out2.creator == f2
    assert out2.creator.inputs == (out1,)
    assert out1.creator == f1
    assert out1.creator.inputs == (test_input,)
