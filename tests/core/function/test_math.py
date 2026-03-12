import numpy as np
import pytest

from ragnarok.core.function import (
    FunctionVariableError,
)
from ragnarok.core.function.common import Function
from ragnarok.core.function.math import (
    Square,
    Exp,
    Add,
    Subtract,
    Multiply,
    Divide,
    Pow,
    Negative,
    Sin,
    Cos,
    Log,
    MatMul,
)
from ragnarok.core.tensor import Tensor
from ragnarok.core.util import numerical_diff, allclose


class FunctionForTest(Function):
    def backward(self, *douts: Tensor):
        return douts[0]

    def forward(self, *variables: Tensor, **kwargs):
        return variables[0]

    def _validate_variables(self, *variables: Tensor):
        pass


class TestSquare:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[2.0, 4.0, 6.0]])),
            ),
            (Tensor(3), Tensor(6)),
            (Tensor(3.0), Tensor(6.0)),
        ],
    )
    def test_backward(self, test_input, expected):
        f = Square()
        f(test_input)
        actual = f.backward(Tensor(1.0))
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[1.0, 4.0, 9.0]])),
            ),
            (Tensor(3), Tensor(9)),
            (Tensor(3.0), Tensor(9.0)),
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
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[1.0, 4.0, 9.0]])),
            ],
            [],
        ],
    )
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Square()
            f(*test_input)

    def test_gradient_check(self):
        test_input = Tensor(np.array([[1.0, 2.0, 3.0]]))
        f1 = Square()
        f2 = Square()

        f1(test_input)
        actual = f1.backward(Tensor(1.0))
        expected = numerical_diff(f2, test_input)
        assert allclose(actual, expected)


class TestExp:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[2.718281828459, 7.389056098931, 20.085536923188]])),
            ),
            (Tensor(3), Tensor(20.085536923188)),
            (Tensor(3.0), Tensor(20.085536923188)),
        ],
    )
    def test_backward(self, test_input, expected):
        f = Exp()
        for_weak_ref = f(test_input)
        actual = f.backward(Tensor(1.0))
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[2.718281828459, 7.389056098931, 20.085536923188]])),
            ),
            (Tensor(3), Tensor(20.085536923188)),
            (Tensor(3.0), Tensor(20.085536923188)),
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
                Tensor(np.array([[1.0, 2.0, 3.0]])),
                Tensor(np.array([[1.0, 4.0, 9.0]])),
            ],
            [],
        ],
    )
    def test__validate_variables(self, test_input):
        with pytest.raises(FunctionVariableError):
            f = Exp()
            f(*test_input)

    def test_gradient_check(self):
        test_input = Tensor(np.array([[1.0, 2.0, 3.0]]))
        f1 = Exp()
        f2 = Exp()

        for_weak_ref = f1(test_input)
        actual = f1.backward(Tensor(1.0))
        expected = numerical_diff(f2, test_input)
        assert allclose(actual, expected)


class TestAdd:
    def test_forward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Add()

        expected = Tensor(np.array([0.4, 0.6]))
        actual = f.forward(test_input1, test_input2)

        assert np.allclose(actual.data, expected.data)

    @pytest.mark.parametrize(
        "shape1, shape2 ,dout, expected1, expected2",
        [
            # same shape case
            (
                (2,),
                (2,),
                Tensor(np.array([1.0, 2.0])),
                Tensor(np.array([1.0, 2.0])),
                Tensor(np.array([1.0, 2.0])),
            ),
            # broadcast case
            (
                (2,),
                (3, 2),
                Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
                Tensor(np.array([6.0, 9.0])),
                Tensor(np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])),
            ),
        ],
    )
    def test_backward(self, shape1, shape2, dout, expected1, expected2):
        var1 = Tensor(np.random.rand(*shape1))
        var2 = Tensor(np.random.rand(*shape2))

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
                Tensor(np.array([1.0, 1.0])),
            ),
            # broadcast case
            (
                (2,),
                (3, 2),
                Tensor(np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])),
            ),
        ],
    )
    def test_gradient_check(self, shape1, shape2, dout):
        f = Add()
        test_inputs = [
            Tensor(np.random.rand(*shape1)),
            Tensor(np.random.rand(*shape2)),
        ]

        f(*test_inputs)

        actual = f.backward(dout)

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestSubtract:
    def test_forward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Subtract()

        expected = Tensor(np.array([-0.2, -0.2]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Subtract()
        f(test_input1, test_input2)
        dout = Tensor(np.array([1.0, 2.0]))

        expected1 = Tensor(np.array([1.0, 2.0]))
        expected2 = Tensor(np.array([-1.0, -2.0]))

        actual1, actual2 = f.backward(dout)

        assert allclose(actual1, expected1)
        assert allclose(actual2, expected2)

    def test_gradient_check(self):
        f = Subtract()
        test_inputs = [
            Tensor(np.array([[3.0, 4.0]])),
            Tensor(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestMultiply:
    def test_forward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Multiply()

        expected = Tensor(np.array([0.03, 0.08]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Multiply()
        f(test_input1, test_input2)
        dout = Tensor(np.array([1.0, 2.0]))

        expected0 = Tensor(np.array([0.3, 0.8]))
        expected1 = Tensor(np.array([0.1, 0.4]))

        dx0, dx1 = f.backward(dout)

        assert allclose(dx0, expected0)
        assert allclose(dx1, expected1)

    def test_gradient_check(self):
        f = Multiply()
        test_inputs = [
            Tensor(np.array([[3.0, 4.0]])),
            Tensor(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestDivide:
    def test_forward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Divide()

        expected = Tensor(np.array([0.1 / 0.3, 0.2 / 0.4]))
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input1 = Tensor(np.array([0.1, 0.2]))
        test_input2 = Tensor(np.array([0.3, 0.4]))

        f = Divide()
        f(test_input1, test_input2)
        dout = Tensor(np.array([1.0, 2.0]))

        expected1 = Tensor(np.array([1.0 / 0.3, 2.0 / 0.4]))
        expected2 = Tensor(np.array([-1.0 * 0.1 / 0.3**2, -2.0 * 0.2 / 0.4**2]))

        actual1, actual2 = f.backward(dout)

        assert allclose(actual1, expected1)
        assert allclose(actual2, expected2)

    def test_gradient_check(self):
        f = Divide()
        test_inputs = [
            Tensor(np.array([[3.0, 4.0]])),
            Tensor(np.array([[6.0, 8.0]])),
        ]

        f(*test_inputs)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, *test_inputs)

        for i, dx in enumerate(actual):
            assert allclose(dx, expected[i])


class TestAutoBroadcast:
    @pytest.mark.parametrize(
        "x0, x1, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                1.0,
                Tensor(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])),
            ),
            (
                1.0,
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])),
            ),
            (
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]])),
            ),
        ],
    )
    def test_add_forward(self, x0, x1, expected):
        f = Add()
        y = f(x0, x1)
        assert allclose(y, expected)

    @pytest.mark.parametrize(
        "x0, x1, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                1.0,
                Tensor(np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])),
            ),
            (
                1.0,
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[0.0, -1.0, -2.0], [-3.0, -4.0, -5.0]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[-9.0, -18.0, -27.0], [-6.0, -15.0, -24.0]])),
            ),
            (
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[9.0, 18.0, 27.0], [6.0, 15.0, 24.0]])),
            ),
        ],
    )
    def test_subtract_forward(self, x0, x1, expected):
        f = Subtract()
        y = f(x0, x1)
        assert allclose(y, expected)

    @pytest.mark.parametrize(
        "x0, x1, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                2.0,
                Tensor(np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])),
            ),
            (
                2.0,
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]])),
            ),
            (
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]])),
            ),
        ],
    )
    def test_multiply_forward(self, x0, x1, expected):
        f = Multiply()
        y = f(x0, x1)
        assert allclose(y, expected)

    @pytest.mark.parametrize(
        "x0, x1, expected",
        [
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                2.0,
                Tensor(np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])),
            ),
            (
                2.0,
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[2.0, 1.0, 0.66666667], [0.5, 0.4, 0.33333333]])),
            ),
            (
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[0.1, 0.1, 0.1], [0.4, 0.25, 0.2]])),
            ),
            (
                Tensor(np.array([10.0, 20.0, 30.0])),
                Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
                Tensor(np.array([[10.0, 10.0, 10.0], [2.5, 4.0, 5.0]])),
            ),
        ],
    )
    def test_divide_forward(self, x0, x1, expected):
        f = Divide()
        y = f(x0, x1)
        assert allclose(y, expected)


class TestNegative:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (Tensor(np.array([0.1, 0.2])), Tensor(np.array([-0.1, -0.2]))),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Negative()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Tensor(np.array([1.1, 1.2]))

        f = Negative()
        f(test_input)
        dout = Tensor(np.array([1.0, 2.0]))

        expected = Tensor(np.array([-1.0, -2.0]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Negative()
        test_input = Tensor(np.array([[1.0, 2.0]]))

        f(test_input)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestPow:
    @pytest.mark.parametrize(
        "test_input,power,expected",
        [
            (Tensor(np.array([0.1, 0.2])), 2, Tensor(np.array([0.01, 0.04]))),
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
                Tensor(np.array([0.1, 0.2])),
                2,
                Tensor(np.array([2 * (0.1**1) * 1.0, 2 * (0.2**1) * 2.0])),
            ),
        ],
    )
    def test_backward(self, test_input, power, expected):
        f = Pow()
        f(test_input, power=power)
        dout = Tensor(np.array([1.0, 2.0]))

        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Pow()
        test_input = Tensor(np.array([[1.0, 2.0]]))

        f(test_input, power=2)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, test_input, power=2)

        assert allclose(actual, expected)


class TestSin:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([0.1, 0.2])),
                Tensor(np.array([np.sin(0.1), np.sin(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Sin()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Tensor(np.array([0.1, 0.2]))

        f = Sin()
        f(test_input)
        dout = Tensor(np.array([1.0, 1.0]))

        expected = Tensor(np.array([np.cos(0.1), np.cos(0.2)]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Sin()
        test_input = Tensor(np.array([[0.1, 0.2]]))

        f(test_input)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestCos:
    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (
                Tensor(np.array([0.1, 0.2])),
                Tensor(np.array([np.cos(0.1), np.cos(0.2)])),
            ),
        ],
    )
    def test_forward(self, test_input, expected):
        f = Cos()
        actual = f.forward(test_input)

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Tensor(np.array([0.1, 0.2]))

        f = Cos()
        f(test_input)
        dout = Tensor(np.array([1.0, 1.0]))

        expected = Tensor(np.array([-np.sin(0.1), -np.sin(0.2)]))
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Cos()
        test_input = Tensor(np.array([[0.1, 0.2]]))

        f(test_input)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestLog:
    def test_forward(self):
        test_input = Tensor([0.1, 0.2])

        f = Log()
        actual = f.forward(test_input)

        expected = Tensor([np.log(0.1), np.log(0.2)])

        assert allclose(actual, expected)

    def test_backward(self):
        test_input = Tensor(np.array([0.1, 0.2]))

        f = Log()
        f(test_input)
        dout = Tensor([1.0, 1.0])

        expected = Tensor([1 / 0.1, 1 / 0.2])
        actual = f.backward(dout)

        assert allclose(actual, expected)

    def test_gradient_check(self):
        f = Log()
        test_input = Tensor(np.array([[0.1, 0.2]]))

        f(test_input)

        actual = f.backward(Tensor(np.array(1.0)))

        expected = numerical_diff(f, test_input)

        assert allclose(actual, expected)


class TestMatMul:
    @pytest.mark.parametrize(
        "test_arr1, test_arr2, expected_arr",
        [
            (
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                [[1.0, 2.0], [2.0, 1.0]],
                [[5.0, 4.0], [8.0, 7.0], [11.0, 10.0]],
            ),
        ],
    )
    def test_forward(self, test_arr1, test_arr2, expected_arr):
        test_input1 = Tensor(test_arr1)
        test_input2 = Tensor(test_arr2)
        expected = Tensor(expected_arr)

        f = MatMul()
        actual = f.forward(test_input1, test_input2)

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_arr1, test_arr2, expected_dx0_arr, expected_dx1_arr",
        [
            (
                [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                [[5.0, 8.0, 11.0], [8.0, 13.0, 18.0]],
                [[5.0, 8.0], [8.0, 13.0], [11.0, 18.0]],
            ),
        ],
    )
    def test_backward(self, test_arr1, test_arr2, expected_dx0_arr, expected_dx1_arr):
        x0 = Tensor(test_arr1)
        x1 = Tensor(test_arr2)

        dout = Tensor([[1.0, 2.0], [2.0, 3.0]])

        f = MatMul()
        f(x0, x1)
        actual_dx0, actual_dx1 = f.backward(dout)

        assert x0.shape == actual_dx0.shape
        assert x1.shape == actual_dx1.shape

        assert allclose(actual_dx0, Tensor(expected_dx0_arr))
        assert allclose(actual_dx1, Tensor(expected_dx1_arr))

    def test_gradient_check(self):
        x0 = Tensor(np.random.rand(2, 3))
        x1 = Tensor(np.random.rand(3, 2))
        dout = Tensor([[1.0, 1.0], [1.0, 1.0]])

        f = MatMul()
        f(x0, x1)

        actual_dx0, actual_dx1 = f.backward(dout)

        expected_dx0, expected_dx1 = numerical_diff(f, x0, x1)

        assert allclose(actual_dx0, expected_dx0)
        assert allclose(actual_dx1, expected_dx1)


def test_define_by_run():
    test_input = Tensor(np.array([0.1, 0.2]))

    f1 = Square()
    f2 = Exp()
    f3 = Square()

    out1 = f1(test_input)
    out2 = f2(out1)
    out3 = f3(out2)

    dout3 = f3.backward(Tensor(1.0))
    dout2 = f2.backward(dout3)
    f1.backward(dout2)

    assert out3.creator == f3
    assert out3.creator.inputs == [out2]
    assert out2.creator == f2
    assert out2.creator.inputs == [out1]
    assert out1.creator == f1
    assert out1.creator.inputs == [test_input]
