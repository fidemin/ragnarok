import numpy as np

from src.main.ragnarok.core.config import using_backprop
from src.main.ragnarok.core.variable import Variable
from src.test.ragnarok.core.function.test_basic_function import FunctionForTest


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
