import math

import numpy as np
import pytest

from core.util import clip_grads


@pytest.mark.parametrize('grads,expected', [
    (
            [np.array([[0.1, 0.2], [0.1, 0.2]]), np.array([[0.1, 0.2]])],
            [np.array([[0.1, 0.2], [0.1, 0.2]]), np.array([[0.1, 0.2]])]
    ),
    (
            [np.array([[3.0, 4.0], [3.0, 4.0]]), np.array([[3.0, 4.0]])],
            [(2.00 / math.sqrt(75.0)) * np.array([[3.0, 4.0], [3.0, 4.0]]),
             (2.00 / math.sqrt(75.0)) * np.array([[3.0, 4.0]])]
    ),
])
def test_clip_grads(grads, expected):
    clip_grads(grads, 2.00)

    for i in range(len(grads)):
        assert np.allclose(grads[i], expected[i])
