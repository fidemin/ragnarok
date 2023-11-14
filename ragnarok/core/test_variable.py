import datetime

import numpy as np
import pytest

from ragnarok.core.function import Square, Exp
from ragnarok.core.variable import Variable, VariableError


class TestVariable:
    @pytest.mark.parametrize('test_input,creator,data', [
        (np.array([[1.0, 2.0, 3.0]]), None, np.array([[1.0, 2.0, 3.0]])),
        (np.array(1), None, np.array(1)),
        (3, None, np.array(3)),
        (3.0, None, np.array(3.0)),
        (np.array([1.0]), Square(), np.array([1.0]))

    ])
    def test_initialization(self, test_input, creator, data):
        variable = Variable(test_input, creator)
        grad = Variable(np.array([1.0, 2.0]))
        variable.grad = grad
        assert np.all(variable.data == data)
        assert variable.creator == creator
        assert variable.grad == grad

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

        # TODO: check gradient for each inputs
