import numpy as np
import pytest

from ragnarok.core.variable import Variable


class TestVariable:
    @pytest.mark.parametrize('test_input', [
        np.array([[1.0, 2.0, 3.0]]),
        3,
        3.0
    ])
    def test_data(self, test_input):
        variable = Variable(test_input)
        assert np.all(test_input == variable.data)
