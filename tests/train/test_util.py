import pytest

from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose
from ragnarok.nn.train.util import accuracy


class TestAccuracy:
    @pytest.mark.parametrize(
        "y, t, expected_acc",
        [
            # All correct
            (
                Tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
                Tensor([1, 0, 1]),
                Tensor(1.0),
            ),
            (
                Tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]),
                Tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
                Tensor(1.0),
            ),
            # All wrong
            (
                Tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
                Tensor([1, 0, 1]),
                Tensor(0.0),
            ),
            (
                Tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]),
                Tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]),
                Tensor(0.0),
            ),
            # Partial correct
            (
                Tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]),
                Tensor([1, 0, 0]),
                Tensor(2 / 3),
            ),
            (
                Tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]),
                Tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]),
                Tensor(2 / 3),
            ),
            # Multiclass
            (
                Tensor(
                    [
                        [0.1, 0.5, 0.4],
                        [0.7, 0.2, 0.1],
                        [0.3, 0.3, 0.4],
                        [0.1, 0.8, 0.1],
                    ]
                ),
                Tensor([1, 0, 2, 0]),
                Tensor(0.75),
            ),
            (
                Tensor(
                    [
                        [0.1, 0.5, 0.4],
                        [0.7, 0.2, 0.1],
                        [0.3, 0.3, 0.4],
                        [0.1, 0.8, 0.1],
                    ]
                ),
                Tensor(
                    [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
                ),
                Tensor(0.75),
            ),
            # Single sample
            (Tensor([[0.2, 0.8]]), Tensor([1]), Tensor(1.0)),
            (Tensor([[0.2, 0.8]]), Tensor([[0.0, 1.0]]), Tensor(1.0)),
        ],
    )
    def test_accuracy(self, y, t, expected_acc):
        acc = accuracy(y, t)
        assert allclose(acc, expected_acc)
