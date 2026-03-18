from ragnarok.core.function.math import MatMul
from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose
from ragnarok.nn.function.activation import relu
from ragnarok.nn.layer.activation import ReLU
from ragnarok.nn.layer.linear import Linear
from ragnarok.nn.model.model import Sequential, MLP


class TestSequential:
    def test_init(self):
        layer1 = Linear(5, 4)
        layer2 = ReLU()
        layer3 = Linear(2, 5, use_bias=False, name="Last")
        layers = [layer1, layer2, layer3]
        model = Sequential(layers)

        model.Linear_1 = layer1
        model.ReLU_2 = layer2
        model.Last_3 = layer3

    def test_forward(self):
        layer1 = Linear(5, 4)
        layer2 = ReLU()
        layer3 = Linear(2, 5, use_bias=False)
        layers = [layer1, layer2, layer3]
        model = Sequential(layers)

        x = Tensor([[1.0, 2.0, -1.0, 3.0], [0.0, -1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0]])
        y = model(x)

        expected = MatMul()(
            relu(MatMul()(x, layer1.W) + layer1.b),
            layer3.W,
        )

        assert y.shape == (3, 2)
        assert allclose(y, expected)


class TestMLP:
    def test_forward(self):
        model = MLP(out_sizes=(8, 4, 2), activation="relu")

        x = Tensor([[1.0, 2.0, -1.0, 3.0], [0.0, -1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0]])
        y = model(x)

        expected = (
            MatMul()(
                relu(
                    MatMul()(
                        relu(MatMul()(x, model.Linear_1.W) + model.Linear_1.b),
                        model.Linear_2.W,
                    )
                    + model.Linear_2.b
                ),
                model.Linear_3.W,
            )
            + model.Linear_3.b
        )

        assert y.shape == (3, 2)
        assert allclose(y, expected)
