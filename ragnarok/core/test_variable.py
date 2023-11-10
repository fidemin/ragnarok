import datetime

import numpy as np
import pytest

from ragnarok.core.function import Square
from ragnarok.core.variable import Variable, VariableError


class TestVariable:
    @pytest.mark.parametrize('test_input,creator', [
        (np.array([[1.0, 2.0, 3.0]]), None),
        (np.array(1), None),
        (3, None),
        (3.0, None),
        (np.array([1.0]), Square())

    ])
    def test_initialization(self, test_input, creator):
        variable = Variable(test_input, creator)
        assert np.all(variable.data == test_input)
        assert variable.creator == creator

    @pytest.mark.parametrize('test_input', [
        'string',
        datetime.datetime.now()
    ])
    def test_raise_error_for_wrong_data_type(self, test_input):
        with pytest.raises(VariableError):
            Variable(test_input)
