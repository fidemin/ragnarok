import datetime
import math

import numpy as np
import pytest

from ragnarok.core.function import Square, Exp, Split, Add
from ragnarok.core.util import allclose, numerical_diff
from ragnarok.core.variable import Variable, VariableError


class TestVariable:
    @pytest.mark.parametrize('test_input,data,shape,ndim,dtype,length', [
        (np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]]), (1, 3), 2, 'float64', 1),
        (np.array(1), np.array(1), (), 0, 'int64', 0),
        (3, np.array(3), (), 0, 'int64', 0),
        (3.0, np.array(3.0), (), 0, 'float64', 0),
        (np.array([1.0]), np.array([1.0]), (1,), 1, 'float64', 1)
    ])
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

    @pytest.mark.parametrize('test_input,data,name', [
        (np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]]), 'name1'),
        (np.array(1), np.array(1), 'name2'),
        (3, np.array(3), 'name3'),
        (3.0, np.array(3.0), 'name4'),
        (np.array([1.0]), np.array([1.0]), 'name5')
    ])
    def test_initialization_with_name(self, test_input, data, name):
        variable = Variable(test_input, name)
        assert np.all(variable.data == data)
        assert variable._name == name

    @pytest.mark.parametrize('test_input', [
        'string',
        datetime.datetime.now()
    ])
    def test_raise_error_for_wrong_data_type(self, test_input):
        with pytest.raises(VariableError):
            Variable(test_input)

    def test_backward(self):
        test_input = Variable(np.array([0.1, 0.2]))

        f1 = Square()
        f2 = Exp()
        f3 = Square()

        out1 = f1(test_input)
        out2 = f2(out1)
        out3 = f3(out2)

        out3.backward()

        test_input_derivative = 2 * np.exp(np.square(test_input.data)) * np.exp(
            np.square(test_input.data)) * 2 * test_input.data
        out1_derivative = 2 * np.exp(np.square(test_input.data)) * np.exp(np.square(test_input.data))
        out2_derivative = 2 * np.exp(np.square(test_input.data))
        assert np.allclose(test_input.grad.data, test_input_derivative)
        assert np.allclose(out1.grad.data, out1_derivative)
        assert np.allclose(out2.grad.data, out2_derivative)

    def test_backward_complex(self):
        """
        function graph
                  Exp
        Split <         > Add
                 Square
        """
        test_input = Variable(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))
        expected = Variable(np.array([[math.exp(0.1), math.exp(0.2), math.exp(0.3)], [0.8, 1.0, 1.2]]))
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

    def test__mult__(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        expected = Variable(np.array([0.03, 0.08]))
        actual = test_input1 * test_input2

        assert allclose(actual, expected)

    def test__mult__backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        forward_result = test_input1 * test_input2

        expected1 = Variable(np.array([0.3, 0.4]))
        expected2 = Variable(np.array([0.1, 0.2]))

        forward_result.backward()

        assert allclose(test_input1.grad, expected1)
        assert allclose(test_input2.grad, expected2)

    def test__add__(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        actual = test_input1 + test_input2

        expected = Variable(np.array([0.4, 0.6]))

        assert allclose(actual, expected)

    def test_add__backward(self):
        test_input1 = Variable(np.array([0.1, 0.2]))
        test_input2 = Variable(np.array([0.3, 0.4]))

        forward_result = test_input1 + test_input2

        expected1 = Variable(np.array([1.0, 1.0]))
        expected2 = Variable(np.array([1.0, 1.0]))

        forward_result.backward()

        assert allclose(test_input1.grad, expected1)
        assert allclose(test_input2.grad, expected2)
