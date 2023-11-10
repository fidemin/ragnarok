import datetime

import numpy as np
import pytest

from ragnarok.core.variable import Variable, VariableError


class TestVariable:
    @pytest.mark.parametrize('test_input', [
        np.array([[1.0, 2.0, 3.0]]),
        3,
        3.0
    ])
    def test_data(self, test_input):
        variable = Variable(test_input)
        assert np.all(test_input == variable.data)

    @pytest.mark.parametrize('test_input', [
        'string',
        datetime.datetime.now()
    ])
    def test_raise_error_for_wrong_data_type(self, test_input):
        with pytest.raises(VariableError):
            variable = Variable(test_input)
