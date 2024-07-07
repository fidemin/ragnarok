import numpy as np
import pytest

from src.main.ragnarok.core.config import using_backprop
from src.main.ragnarok.core.function import (
    Square,
    FunctionVariableError,
    Exp,
    Add,
    Split,
    Function,
    Multiply,
    Negative,
    Subtract,
    Divide,
    Pow,
    Sin,
    Cos,
    Tanh,
    Reshape,
    Transpose,
    SumTo,
    BroadcastTo,
)
from src.main.ragnarok.core.util import numerical_diff, allclose
from src.main.ragnarok.core.variable import Variable


class FunctionForTest(Function):
    def backward(self, *douts: Variable):
        return douts[0]

    def forward(self, *variables: Variable, **kwargs):
        return variables[0]

    def _validate_variables(self, *variables: Variable):
        pass


class TestFunction:
    def test_call(self):
        # generation of test_input is 0
        test_input = Variable(np.array([1.0, 2.0, 3.0]))
        f = FunctionForTest()
        output = f(test_input)

        assert output.gen == 1
        assert f.gen == 0
        assert output.creator == f

    def test_call_with_using_backprop_false(self):
        test_input = Variable(np.array([1.0, 2.0, 3.0]))
        with using_backprop(False):
            f = FunctionForTest()
            output = f(test_input)

        assert f.gen is None
        assert f.inputs is None
        assert f.outputs is None

    def test_call__using_float_int(self):
        f = FunctionForTest()
        output = f(3.0, 4)
        for input_ in f.inputs:
            assert isinstance(input_, Variable)
        assert f.inputs[0].data == 3.0
        assert f.inputs[1].data == 4


class TestSquare:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.0, 4.0, 6.0]])),
            ),
            (Variable(3), Variable(6)),
            (Variable(3.0), Variable(6.0)),
        ],
    )
    def test_backward(self, test_input, expected):
        f = Square()
        f(test_input)
        actual = f.backward(Variable(1.0))
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[1.0, 4.0, 9.0]])),
            ),
            (Variable(3), Variable(9)),
            (Variable(3.0), Variable(9.0)),
        ],
    )
    def test_call(self, test_input, expected):
        f = Square()
        actual = f(test_input)
        assert actual.creator is f
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input",
        [
            [
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[1.0, 4.0, 9.0]])),
            ],
            [],
        ],
    )
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
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.718281828459, 7.389056098931, 20.085536923188]])),
            ),
            (Variable(3), Variable(20.085536923188)),
            (Variable(3.0), Variable(20.085536923188)),
        ],
    )
    def test_backward(self, test_input, expected):
        f = Exp()
        for_weak_ref = f(test_input)
        actual = f.backward(Variable(1.0))
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[2.718281828459, 7.389056098931, 20.085536923188]])),
            ),
            (Variable(3), Variable(20.085536923188)),
            (Variable(3.0), Variable(20.085536923188)),
        ],
    )
    def test_call(self, test_input, expected):
        f = Exp()
        actual = f(test_input)
        assert actual.creator is f
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input",
        [
            [
                Variable(np.array([[1.0, 2.0, 3.0]])),
                Variable(np.array([[1.0, 4.0, 9.0]])),
            ],
            [],
        ],
    )
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Exp()
            f(*test_input)

    def test_gradient_check(self):
        test_input = Variable(np.array([[1.0, 2.0, 3.0]]))
        f1 = Exp()
        f2 = Exp()

        for_weak_ref = f1(test_input)
        actual = f1.backward(Variable(1.0))
        expected = numerical_diff(f2, test_input)
        assert allclose(actual, expected)


class TestAdd:
    def test_forward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Add()

        expected = Variable(np.array([0.4, 0.6]))
        actual = f.forward(test_input1, test_input2)

        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize(
        "shape1, shape2 ,dout, expected1, expected2",
        [
            # same shape case
            (
                (2,),
                (2,),
                Variable(np.array([1.0, 2.0])),
                Variable(np.array([1.0, 2.0])),
                Variable(np.array([1.0, 2.0])),
            ),
            # broadcast case
            (
                (2,),
                (3, 2),
                Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                Variable(np.array([6.0, 9.0])),
                Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, shape1, shape2, dout, expected1, expected2):
        var1 = Variable(np.random.rand(*shape1))
        var2 = Variable(np.random.rand(*shape2))

        f = Add()
        f(var1, var2)

        dx0, dx1 = f.backward(dout)

        assert np.allclose(dx0.data, expected1.data)
        assert np.allclose(dx1.data, expected2.data)

    @pytest.mark.parametrize(
        "shape1, shape2 ,dout",
        [
            # same shape case
            (
                (2,),
                (2,),
                Variable(np.array([1.0, 1.0])),
            ),
            # broadcast case
            (
                (2,),
                (3, 2),
                Variable(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])),
            ),
        ],
    )
    def test_gradient_check(self, shape1, shape2, dout):
        f = Add()
        test_inputs = [
            Variable(np.random.rand(*shape1)),
            Variable(np.random.rand(*shape2)),
        ]

        f(*test_inputs)

        actual = f.backward(dout)

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestSubtract:
    def test_forward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Subtract()

        expected = Variable(np.array([-0.2, -0.2]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Subtract()
        f(test_input1, test_input2)
        dout = Variable(np.array([1.0, 2.0]))

        expected1 = Variable(np.array([1.0, 2.0]))
        expected2 = Variable(np.array([-1.0, -2.0]))

        actual1, actual2 = f.backward(dout)

        assert allclose(actual1, expected1)
        assert allclose(actual2, expected2)

    def test_gradient_check(self):
        f = Subtract()
        test_inputs = [
            Variable(np.array([[3.0, 4.0]])),
            Variable(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestMultiply:
    def test_forward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Multiply()

        expected = Variable(np.array([0.03, 0.08]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Multiply()
        f(test_input1, test_input2)
        dout = Variable(np.array([1.0, 2.0]))

        expected0 = Variable(np.array([0.3, 0.8]))
        expected1 = Variable(np.array([0.1, 0.4]))

        dx0, dx1 = f.backward(dout)

        assert allclose(dx0, expected0)
        assert allclose(dx1, expected1)

    def test_gradient_check(self):
        f = Multiply()
        test_inputs = [
            Variable(np.array([[3.0, 4.0]])),
            Variable(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestDivide:
    def test_forward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Divide()

        expected = Variable(np.array([0.1 / 0.3, 0.2 / 0.4]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        f = Divide()
        f(test_input1, test_input2)
        dout = Variable(np.array([1.0, 2.0]))

        expected1 = Variable(np.array([1.0 / 0.3, 2.0 / 0.4]))
        expected2 = Variable(np.array([-1.0 * 0.1 / 0.3**2, -2.0 * 0.2 / 0.4**2]))

        actual1, actual2 = f.backward(dout)

        assert allclose(actual1, expected1)
        assert allclose(actual2, expected2)

    def test_gradient_check(self):
        f = Divide()
        test_inputs = [
            Variable(np.array([[3.0, 4.0]])),
            Variable(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestNegative:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (Variable(np.array([0.1, 0.2])), Variable(np.array([-0.1, -0.2]))),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Negative()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Variable(np.array([1.1, 1.2]))

        f = Negative()
        f(test_input)
        dout = Variable(np.array([1.0, 2.0]))

        expected = Variable(np.array([-1.0, -2.0]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Negative()
        test_input = Variable(np.array([[1.0, 2.0]]))

        f(test_input)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestPow:
    @pytest.mark.parametrize(
        "test_input,power,expected",
        [
            (Variable(np.array([0.1, 0.2])), 2, Variable(np.array([0.01, 0.04]))),
        ],
    )
    def test_forward(self, test_input, power, expected):
        f = Pow()
        actual = f.forward(test_input, power=power)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,power,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                2,
                Variable(np.array([2 * (0.1**1) * 1.0, 2 * (0.2**1) * 2.0])),
            ),
        ],
    )
    def test_backward(self, test_input, power, expected):
        f = Pow()
        f(test_input, power=power)
        dout = Variable(np.array([1.0, 2.0]))

        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Pow()
        test_input = Variable(np.array([[1.0, 2.0]]))

        f(test_input, power=2)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input, power=2)

        assert allclose(actual, expected)


class TestSin:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([np.sin(0.1), np.sin(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Sin()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f = Sin()
        f(test_input)
        dout = Variable(np.array([1.0, 1.0]))

        expected = Variable(np.array([np.cos(0.1), np.cos(0.2)]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Sin()
        test_input = Variable(np.array([[0.1, 0.2]]))

        f(test_input)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestCos:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([np.cos(0.1), np.cos(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Cos()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f = Cos()
        f(test_input)
        dout = Variable(np.array([1.0, 1.0]))

        expected = Variable(np.array([-np.sin(0.1), -np.sin(0.2)]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Cos()
        test_input = Variable(np.array([[0.1, 0.2]]))

        f(test_input)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestTanh:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([np.tanh(0.1), np.tanh(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Tanh()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f = Tanh()
        y_for_weak_ref = f(test_input)
        dout = Variable(np.array([1.0, 1.0]))

        expected = Variable(np.array([1 - np.tanh(0.1) ** 2, 1 - np.tanh(0.2) ** 2]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Tanh()
        test_input = Variable(np.array([[0.1, 0.2]]))

        y_for_weak_ref = f(test_input)

        actual = f.backward(Variable(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestSplit:
    @pytest.mark.parametrize(
        "test_input,axis,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                0,
                [
                    Variable(np.array([[1.0, 2.0, 3.0]])),
                    Variable(np.array([[2.0, 4.0, 8.0]])),
                ],
            ),
            (
                Variable(np.array([[1.0, 3.0], [2.0, 4.0]])),
                1,
                [
                    Variable(np.array([[1.0], [2.0]])),
                    Variable(np.array([[3.0], [4.0]])),
                ],
            ),
        ],
    )
    def test_forward(self, test_input, axis, expected):
        split = Split()
        actual = split.forward(test_input, axis=axis)
        assert len(actual) == len(expected)

        for i in range(len(actual)):
            assert allclose(actual[i], expected[i])

    @pytest.mark.parametrize(
        "test_input,axis,expected",
        [
            (
                [
                    Variable(np.array([[1.0, 2.0, 3.0]])),
                    Variable(np.array([[2.0, 4.0, 8.0]])),
                ],
                0,
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                [
                    Variable(np.array([[1.0], [2.0]])),
                    Variable(np.array([[3.0], [4.0]])),
                ],
                1,
                Variable(np.array([[1.0, 3.0], [2.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, test_input, axis, expected):
        output_shape = expected.data.shape
        forward_input = Variable(np.random.rand(*output_shape))

        split = Split()
        split(forward_input, axis=axis)
        actual = split.backward(*test_input)

        assert allclose(actual, expected)


class TestReshape:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                (3, 2),
                Variable(np.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]])),
            ),
            (
                Variable(np.array([[1.0, 3.0], [2.0, 4.0]])),
                (4,),
                Variable(np.array([1.0, 3.0, 2.0, 4.0])),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        split = Reshape()
        actual = split.forward(test_input, shape=shape)
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            (
                Variable(np.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]])),
                (3, 2),
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                Variable(np.array([1.0, 3.0, 2.0, 4.0])),
                (2, 2),
                Variable(np.array([[1.0, 3.0], [2.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Variable(np.random.rand(*output_shape))

        split = Reshape()
        split(forward_input, shape=shape)
        actual = split.backward(test_input)

        assert allclose(actual, expected)


class TestTranspose:
    @pytest.mark.parametrize(
        "test_input,transpose, expected",
        [
            (
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                None,
                Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
            ),
            (
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
                (1, 0),
                Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
            ),
            # more complex case
            (
                Variable(
                    np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
                ),
                (1, 0, 2),
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [5.0, 6.0]],
                            [[3.0, 4.0], [7.0, 8.0]],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, transpose, expected):
        f = Transpose()
        if transpose:
            actual = f.forward(test_input, transpose=transpose)
        else:
            actual = f.forward(test_input)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,transpose, expected",
        [
            (
                Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                None,
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            (
                Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                (1, 0),
                Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])),
            ),
            # more complex case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [5.0, 6.0]],
                            [[3.0, 4.0], [7.0, 8.0]],
                        ]
                    )
                ),
                (1, 0, 2),
                Variable(
                    np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
                ),
            ),
        ],
    )
    def test_backward(self, test_input, transpose, expected):
        output_shape = expected.shape
        forward_input = Variable(np.random.rand(*output_shape))

        f = Transpose()
        if transpose:
            f(forward_input, transpose=transpose)
        else:
            f(forward_input)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,transpose",
        [
            # multi inputs
            (
                [
                    Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])),
                    Variable(1.0),
                ],
                (1, 0, 2),
            ),
            # transpose is not tuple
            (
                [Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]))],
                1,
            ),
        ],
    )
    def test_validate_variables(self, test_input, transpose):
        with pytest.raises(FunctionVariableError):
            f = Transpose()(*test_input, transpose=transpose)


class TestBroadcastTo:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no broadcast required
            (
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) case
            (
                Variable(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Variable(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2) -> (2, 3, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0], [6.0, 9.0], [6.0, 9.0]],
                            [[15.0, 18.0], [15.0, 18.0], [15.0, 18.0]],
                        ]
                    )
                ),
            ),
            #  (2, 1, 2, 1) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                            ],
                            [
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[12.0, 16.0]],
                            [[16.0, 20.0]],
                            [[20.0, 24.0]],
                        ],
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        f = BroadcastTo()
        actual = f.forward(test_input, shape=shape)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "from_shape, to_shape",
        [
            (
                (1, 2, 3),
                (2, 3),
            ),
            (
                (3, 3),
                (2, 3, 2),
            ),
            (
                (1, 3, 3),
                (2, 3, 2),
            ),
        ],
    )
    def test_forward_error(self, from_shape, to_shape):
        with pytest.raises(FunctionVariableError) as exc_info:
            f = BroadcastTo()
            f.forward(Variable(np.random.rand(*from_shape)), shape=to_shape)

        print("error message: ", exc_info.value)

    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no change
            (
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
            ),
            # (2, 1, 2) -> (2, 3, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 1) -> (2, 3, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[3.0], [5.0], [7.0]],
                            [[9.0], [11.0], [13.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2, 1) -> (2, 3, 2, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) broadcast case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[12.0, 16.0]],
                                [[16.0, 20.0]],
                                [[20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Variable(np.random.rand(*output_shape))

        f = BroadcastTo()
        f(forward_input, shape=shape)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = BroadcastTo()
        shape = (3, 2)
        test_input = Variable(np.random.rand(*shape))

        f(test_input, shape=(2, 3, 2))

        variable = Variable(
            np.array(
                [
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                ]
            )
        )

        actual = f.backward(variable)

        expected = numerical_diff(f, test_input, shape=(2, 3, 2))

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, shape",
        [
            # no shape
            ([Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], None),
            # shape is not tuple
            ([Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], 3),
            # multiple variables
            (
                [
                    Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                    Variable(np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])),
                ],
                (3, 2),
            ),
        ],
    )
    def test_validate_variables(self, test_input, shape):
        with pytest.raises(FunctionVariableError) as exc_info:
            f = BroadcastTo()
            if shape:
                f(*test_input, shape=shape)
            else:
                f(*test_input)

        print("error message: ", exc_info.value)


class TestSumTo:
    @pytest.mark.parametrize(
        "test_input,shape,expected",
        [
            # (3, 2) -> (3, 2) case: no sum required
            (
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (2, 3, 2) -> (3, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (3, 2),
                Variable(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
            ),
            # (2, 3, 2) -> (1, 3, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (1, 3, 2),
                Variable(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
            ),
            # (2, 3, 2) -> (2, 1, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 1, 2),
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 2) -> (2, 3, 1) case
            (
                Variable(
                    np.array(
                        [
                            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                            [[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
                        ]
                    )
                ),
                (2, 3, 1),
                Variable(
                    np.array(
                        [
                            [[3.0], [5.0], [7.0]],
                            [[9.0], [11.0], [13.0]],
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (2, 1, 2, 1) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (2, 1, 2, 1),
                Variable(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (1, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (1, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
            ),
            # (2, 3, 2, 2) -> (3, 1, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[1.0, 2.0], [2.0, 3.0]],
                                [[2.0, 3.0], [3.0, 4.0]],
                                [[3.0, 4.0], [4.0, 5.0]],
                            ],
                            [
                                [[4.0, 5.0], [5.0, 6.0]],
                                [[5.0, 6.0], [6.0, 7.0]],
                                [[6.0, 7.0], [7.0, 8.0]],
                            ],
                        ]
                    )
                ),
                (3, 1, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[12.0, 16.0]],
                                [[16.0, 20.0]],
                                [[20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_forward(self, test_input, shape, expected):
        f = SumTo()
        actual = f.forward(test_input, shape=shape)
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "from_shape, to_shape",
        [
            # The length of the given shape is larger than the length of the input
            (
                (2, 3),
                (1, 2, 3),
            ),
            # The from_shape can not sum to to_shape
            (
                (2, 3, 2),
                (3, 3),
            ),
            (
                (2, 3, 2),
                (1, 3, 3),
            ),
        ],
    )
    def test_forward_error(self, from_shape, to_shape):
        test_input = Variable(np.random.rand(*from_shape))
        with pytest.raises(FunctionVariableError) as exc_info:
            f = SumTo()
            f.forward(test_input, shape=to_shape)

        print("error message: ", exc_info.value)

    @pytest.mark.parametrize(
        "test_input, shape, expected",
        [
            # (3, 2) -> (3, 2) case: no broadcast required
            (
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
                (3, 2),
                Variable(
                    np.array(
                        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                    )
                ),
            ),
            # (3, 2) -> (2, 3, 2) case
            (
                Variable(np.array([[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]])),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (1, 3, 2) -> (2, 3, 2) case
            (
                Variable(np.array([[[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]]])),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                            [[5.0, 7.0], [7.0, 9.0], [9.0, 11.0]],
                        ]
                    )
                ),
            ),
            # (2, 1, 2) -> (2, 3, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0]],
                            [[15.0, 18.0]],
                        ]
                    )
                ),
                (2, 3, 2),
                Variable(
                    np.array(
                        [
                            [[6.0, 9.0], [6.0, 9.0], [6.0, 9.0]],
                            [[15.0, 18.0], [15.0, 18.0], [15.0, 18.0]],
                        ]
                    )
                ),
            ),
            #  (2, 1, 2, 1) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[15.0], [21.0]],
                            ],
                            [
                                [[33.0], [39.0]],
                            ],
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                                [[15.0, 15.0], [21.0, 21.0]],
                            ],
                            [
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                                [[33.0, 33.0], [39.0, 39.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (1, 2, 2) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                            ]
                        ]
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                            [
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                                [[21.0, 27.0], [27.0, 33.0]],
                            ],
                        ]
                    )
                ),
            ),
            # (3, 1, 2) -> (2, 3, 2, 2) case
            (
                Variable(
                    np.array(
                        [
                            [[12.0, 16.0]],
                            [[16.0, 20.0]],
                            [[20.0, 24.0]],
                        ],
                    )
                ),
                (2, 3, 2, 2),
                Variable(
                    np.array(
                        [
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                            [
                                [[12.0, 16.0], [12.0, 16.0]],
                                [[16.0, 20.0], [16.0, 20.0]],
                                [[20.0, 24.0], [20.0, 24.0]],
                            ],
                        ]
                    )
                ),
            ),
        ],
    )
    def test_backward(self, test_input, shape, expected):
        output_shape = expected.data.shape
        forward_input = Variable(np.random.rand(*output_shape))

        f = SumTo()
        f(forward_input, shape=shape)
        actual = f.backward(test_input)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = SumTo()
        shape = (2, 3, 2)
        test_input = Variable(np.random.rand(*shape))

        f(test_input, shape=(3, 2))

        variable = Variable(
            np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            )
        )

        actual = f.backward(variable)

        expected = numerical_diff(f, test_input, shape=(3, 2))

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input, shape",
        [
            # no shape
            ([Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], None),
            # shape is not tuple
            ([Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))], 3),
            # multiple variables
            (
                [
                    Variable(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                    Variable(np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])),
                ],
                (3, 2),
            ),
        ],
    )
    def test_validate_variables(self, test_input, shape):
        with pytest.raises(FunctionVariableError) as exc_info:
            f = SumTo()
            if shape:
                f(*test_input, shape=shape)
            else:
                f(*test_input)

        print("error message: ", exc_info.value)


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
    f1.backward(dout2)

    assert out3.creator == f3
    assert out3.creator.inputs == [out2]
    assert out2.creator == f2
    assert out2.creator.inputs == [out1]
    assert out1.creator == f1
    assert out1.creator.inputs == [test_input]
