import datetime
import math

import numpy as np
import pytest

from src.main.ragnarok.core.function import Square, Exp, Split, Add
from src.main.ragnarok.core.util import allclose, numerical_diff
from src.main.ragnarok.core.variable import Variable, VariableError, to_variable
from src.main.ragnarok.core.variable.dtype import int8


class TestVariable:
    @pytest.mark.parametrize(
        "test_input,data,shape,ndim,dtype,length",
        [
            (
                np.array([[1.0, 2.0, 3.0]]),
                np.array([[1.0, 2.0, 3.0]]),
                (1, 3),
                2,
                "float64",
                1,
            ),
            (np.array(1), np.array(1), (), 0, "int64", 0),
            (3, np.array(3), (), 0, "int64", 0),
            (3.0, np.array(3.0), (), 0, "float64", 0),
            (np.array([1.0]), np.array([1.0]), (1,), 1, "float64", 1),
            (
                [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
                (3, 2),
                2,
                "float64",
                3,
            ),
        ],
    )
    def test_initialization(self, test_input, data, shape, ndim, dtype, length):
        variable = Variable(test_input)
        grad = Variable(np.array([1.0, 2.0]))
        variable.grad = grad
        assert np.all(variable.data == data)
        assert variable.grad == grad
        assert variable.shape == shape
        assert variable._name is None
        assert variable.ndim == ndim
        assert variable.dtype == dtype
        assert len(variable) == length

    @pytest.mark.parametrize(
        "test_input,data,name",
        [
            (np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]]), "name1"),
            (np.array(1), np.array(1), "name2"),
            (3, np.array(3), "name3"),
            (3.0, np.array(3.0), "name4"),
            (np.array([1.0]), np.array([1.0]), "name5"),
        ],
    )
    def test_initialization_with_name(self, test_input, data, name):
        variable = Variable(test_input, name)
        assert np.all(variable.data == data)
        assert variable._name == name

    @pytest.mark.parametrize("test_input", ["string", datetime.datetime.now()])
    def test_raise_error_for_wrong_data_type(self, test_input):
        with pytest.raises(VariableError):
            Variable(test_input)

    def test_astype(self):
        variable = Variable([True, False, True])
        actual = variable.astype(int8)
        expected = Variable(np.array([1, 0, 1], dtype=int8))
        assert allclose(actual, expected)
        assert actual.dtype == expected.dtype

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f1 = Square()
        f2 = Exp()
        f3 = Square()

        out1 = f1(test_input)
        out2 = f2(out1)
        out3 = f3(out2)

        out3.backward(retain_grad=True)

        test_input_derivative = (
            2
            * np.exp(np.square(test_input.data))
            * np.exp(np.square(test_input.data))
            * 2
            * test_input.data
        )
        out1_derivative = (
            2 * np.exp(np.square(test_input.data)) * np.exp(np.square(test_input.data))
        )
        out2_derivative = 2 * np.exp(np.square(test_input.data))
        assert np.allclose(test_input.grad.data, test_input_derivative)
        assert np.allclose(out1.grad.data, out1_derivative)
        assert np.allclose(out2.grad.data, out2_derivative)

    def test_backward__with_no_retain_grad(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f1 = Square()
        f2 = Exp()
        f3 = Square()

        out1 = f1(test_input)
        out2 = f2(out1)
        out3 = f3(out2)

        out3.backward()

        assert out1.grad is None
        assert out2.grad is None
        assert test_input.grad is not None

    def test_backward_complex(self):
        """
        function graph
                  Exp
        Split <         > Add
                 Square
        """
        test_input = Variable(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        expected = Variable(
            np.array([[math.exp(0.1), math.exp(0.2), math.exp(0.3)], [0.8, 1.0, 1.2]])
        )
        f1 = Split()
        f2_1 = Exp()
        f2_2 = Square()
        f3 = Add()

        out1_1, out1_2 = f1(test_input, axis=0)
        out2_1 = f2_1(out1_1)
        out2_2 = f2_2(out1_2)
        out = f3(out2_1, out2_2)
        out.backward()

        assert allclose(test_input.grad, expected)

    def test_backward_complex_with_gradient_check(self):
        test_input = Variable(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        f1 = Split()
        f2_1 = Exp()
        f2_2 = Square()
        f3 = Add()

        out1_1, out1_2 = f1(test_input, axis=0)
        out2_1 = f2_1(out1_1)
        out2_2 = f2_2(out1_2)
        out = f3(out2_1, out2_2)
        out.backward()

        def complex_function(*variables):
            nout1_1, nout1_2 = f1(*variables, axis=0)
            nout2_1 = f2_1(nout1_1)
            nout2_2 = f2_2(nout1_2)
            return f3(nout2_1, nout2_2)

        expected = numerical_diff(complex_function, test_input)

        assert allclose(test_input.grad, expected)

    def test_backward_with_same_inputs(self):
        test_input = Variable(np.array([0.1, 0.2, 0.3]))
        f1 = Add()
        f2 = Add()
        output1 = f1(test_input, test_input)
        output = f2(output1, test_input)

        output.backward()

        expected = Variable(np.array([3.0, 3.0, 3.0]))
        assert allclose(test_input.grad, expected)

    def test_set_creator(self):
        test_input = Variable(np.array([0.1, 0.2]))
        func = Square()
        func.gen = test_input.gen
        test_input.set_creator(func)

        assert test_input.creator == func
        assert test_input.gen == 1

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.03, 0.08])),
            ),
            (Variable(np.array([0.1, 0.2])), 0.3, Variable(np.array([0.03, 0.06]))),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([0.03, 0.06])),
            ),
            (0.1, Variable(np.array([0.3, 0.4])), Variable(np.array([0.03, 0.04]))),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.03, 0.04])),
            ),
        ],
    )
    def test__mult__(self, test_input1, test_input2, expected):
        actual = test_input1 * test_input2

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected1,expected2",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.1, 0.2])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                0.3,
                Variable(0.3),
                Variable(np.array([0.1, 0.2])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(0.3),
                Variable(np.array([0.1, 0.2])),
            ),
            (
                0.1,
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.3, 0.4])),
                Variable(0.1),
            ),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.3, 0.4])),
                Variable(0.1),
            ),
        ],
    )
    def test__mult__backward(self, test_input1, test_input2, expected1, expected2):
        forward_result = test_input1 * test_input2

        forward_result.backward()
        inputs = forward_result.creator.inputs

        assert allclose(inputs[0].grad, expected1)
        assert allclose(inputs[1].grad, expected2)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.1 / 0.3, 0.2 / 0.4])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                0.3,
                Variable(np.array([0.1 / 0.3, 0.2 / 0.3])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([0.1 / 0.3, 0.2 / 0.3])),
            ),
            (
                0.1,
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.1 / 0.3, 0.1 / 0.4])),
            ),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.1 / 0.3, 0.1 / 0.4])),
            ),
        ],
    )
    def test__true_div__(self, test_input1, test_input2, expected):
        actual = test_input1 / test_input2

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected1,expected2",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([1.0 / 0.3, 1.0 / 0.4])),
                Variable(np.array([-0.1 / 0.3**2, -0.2 / 0.4**2])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                0.3,
                Variable(np.array([1.0 / 0.3, 1.0 / 0.3])),
                Variable(np.array([-0.1 / 0.3**2, -0.2 / 0.3**2])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([1.0 / 0.3, 1.0 / 0.3])),
                Variable(np.array([-0.1 / 0.3**2, -0.2 / 0.3**2])),
            ),
            (
                0.1,
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([1.0 / 0.3, 1.0 / 0.4])),
                Variable(np.array([-0.1 / 0.3**2, -0.1 / 0.4**2])),
            ),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([1.0 / 0.3, 1.0 / 0.4])),
                Variable(np.array([-0.1 / 0.3**2, -0.1 / 0.4**2])),
            ),
        ],
    )
    def test__true_div_backward(self, test_input1, test_input2, expected1, expected2):
        forward_result = test_input1 / test_input2

        forward_result.backward()
        inputs = forward_result.creator.inputs

        assert allclose(inputs[0].grad, expected1)
        assert allclose(inputs[1].grad, expected2)

    @pytest.mark.parametrize(
        "test_input,expected",
        [
            (Variable(np.array([0.1, 0.2])), Variable(np.array([-0.1, -0.2]))),
            (np.array([0.1, 0.2]), Variable(np.array([-0.1, -0.2]))),
        ],
    )
    def test__neg__(self, test_input, expected):
        actual = -test_input

        assert allclose(actual, expected)

    def test__pow__(self):
        test_input = Variable(np.array([0.1, 0.2]))
        actual = test_input**2

        expected = Variable(np.array([0.01, 0.04]))
        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.4, 0.6])),
            ),
            (Variable(np.array([0.1, 0.2])), 0.3, Variable(np.array([0.4, 0.5]))),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([0.4, 0.5])),
            ),
            (0.1, Variable(np.array([0.3, 0.4])), Variable(np.array([0.4, 0.5]))),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([0.4, 0.5])),
            ),
        ],
    )
    def test__add__(self, test_input1, test_input2, expected):
        actual = test_input1 + test_input2

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected1,expected2",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([1.0, 1.0])),
                Variable(np.array([1.0, 1.0])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                0.3,
                Variable(np.array([1.0, 1.0])),
                Variable(2.0),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([1.0, 1.0])),
                Variable(2.0),
            ),
            (
                0.1,
                Variable(np.array([0.3, 0.4])),
                Variable(2.0),
                Variable(np.array([1.0, 1.0])),
            ),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([2.0])),
                Variable(np.array([1.0, 1.0])),
            ),
        ],
    )
    def test_add__backward(self, test_input1, test_input2, expected1, expected2):
        forward_result = test_input1 + test_input2

        forward_result.backward()
        inputs = forward_result.creator.inputs

        assert allclose(inputs[0].grad, expected1)
        assert allclose(inputs[1].grad, expected2)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([-0.2, -0.2])),
            ),
            (Variable(np.array([0.1, 0.2])), 0.3, Variable(np.array([-0.2, -0.1]))),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([-0.2, -0.1])),
            ),
            (0.1, Variable(np.array([0.3, 0.4])), Variable(np.array([-0.2, -0.3]))),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([-0.2, -0.3])),
            ),
        ],
    )
    def test_sub__(self, test_input1, test_input2, expected):
        actual = test_input1 - test_input2

        assert allclose(actual, expected)

    @pytest.mark.parametrize(
        "test_input1,test_input2,expected1,expected2",
        [
            (
                Variable(np.array([0.1, 0.2])),
                Variable(np.array([0.3, 0.4])),
                Variable(np.array([1.0, 1.0])),
                Variable(np.array([-1.0, -1.0])),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                0.3,
                Variable(np.array([1.0, 1.0])),
                Variable(-1.0),
            ),
            (
                Variable(np.array([0.1, 0.2])),
                np.array([0.3]),
                Variable(np.array([1.0, 1.0])),
                Variable(-1.0),
            ),
            (
                0.1,
                Variable(np.array([0.3, 0.4])),
                Variable(1.0),
                Variable(np.array([-1.0, -1.0])),
            ),
            (
                np.array([0.1]),
                Variable(np.array([0.3, 0.4])),
                Variable(1.0),
                Variable(np.array([-1.0, -1.0])),
            ),
        ],
    )
    def test_sub__backward(self, test_input1, test_input2, expected1, expected2):
        forward_result = test_input1 - test_input2

        forward_result.backward()
        inputs = forward_result.creator.inputs

        assert allclose(inputs[0].grad, expected1)
        assert allclose(inputs[1].grad, expected2)

    @pytest.mark.parametrize(
        "operator, expected_list",
        [
            ("eq", [False, False, True]),
            ("ne", [True, True, False]),
            ("gt", [False, True, False]),
            ("ge", [False, True, True]),
            ("lt", [True, False, False]),
            ("le", [True, False, True]),
        ],
    )
    def test_built_in_comparison_operations(self, operator, expected_list):
        test_input1 = Variable([0.1, 0.2, 0.3])
        test_input2 = Variable([0.2, 0.1, 0.3])
        expected = Variable(expected_list)

        if operator == "eq":
            actual = test_input1 == test_input2
        elif operator == "ne":
            actual = test_input1 != test_input2
        elif operator == "gt":
            actual = test_input1 > test_input2
        elif operator == "ge":
            actual = test_input1 >= test_input2
        elif operator == "lt":
            actual = test_input1 < test_input2
        elif operator == "le":
            actual = test_input1 <= test_input2
        else:
            raise ValueError("Invalid operator")

        assert allclose(actual, expected)

    def test_release(self):
        test_input = Variable(np.array([0.1, 0.2]))
        test_input.release()

        assert test_input.data is None

    def test_second_order_differentiation(self):
        x = Variable(np.array([2.0]))
        y: Variable = x**4 - x**2 - x
        y.backward(enable_double_backprop=True)
        assert allclose(x.grad, Variable(np.array([27.0])))

        gx = x.grad
        x.clear_grad()
        gx.backward()

        expected = Variable(np.array([46.0]))
        assert allclose(x.grad, expected)

    def test_reshape(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.reshape(3, 2)
        assert y.shape == (3, 2)
        assert allclose(y, Variable(np.array([[1, 2], [3, 4], [5, 6]])))

        y = x.reshape((3, 2))
        assert y.shape == (3, 2)
        assert allclose(y, Variable(np.array([[1, 2], [3, 4], [5, 6]])))

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
    def test_transpose(self, test_input, transpose, expected):
        if not transpose:
            actual = test_input.transpose()
        else:
            actual = test_input.transpose(*transpose)

        assert allclose(actual, expected)

        if transpose:
            actual = test_input.transpose(transpose)

        assert allclose(actual, expected)

    def test_T(self):
        x = Variable(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]]))
        y = x.T
        expected = Variable(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]]))
        assert allclose(y, expected)


@pytest.mark.parametrize(
    "input_value, expected_type",
    [
        (1, Variable),
        (1.0, Variable),
        (np.array([1, 2, 3]), Variable),
        (np.float32(1.0), Variable),
    ],
)
def test_to_variable(input_value, expected_type):
    result = to_variable(input_value)
    assert isinstance(result, expected_type)
    assert np.all(result.data == input_value)


def test_to_variable__from_variable():
    input_value = Variable(np.array([1, 2, 3]))
    result = to_variable(input_value)
    assert result is input_value
