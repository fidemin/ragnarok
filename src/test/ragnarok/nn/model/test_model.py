from src.main.ragnarok.core.function import MatMul
from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import relu
from src.main.ragnarok.nn.layer.activation import ReLU
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Sequential


class TestSequential:
    def test_init(self):
        layer1 = Linear(5, 4)
        layer2 = ReLU()
        layer3 = Linear(2, 5, use_bias=False, name="Last")
        layers = [layer1, layer2, layer3]
        model = Sequential(layers)

        assert model.layers == layers
        assert [layer.name for layer in model.layers] == [
            "Linear__1",
            "ReLU__2",
            "Last__3",
        ]
        assert model.params_dict == {
            "Linear__1__W": layer1.params["W"],
            "Linear__1__b": layer1.params["b"],
            "Last__3__W": layer3.params["W"],
        }

    def test_predict(self):
        layer1 = Linear(5, 4)
        layer2 = ReLU()
        layer3 = Linear(2, 5, use_bias=False)
        layers = [layer1, layer2, layer3]
        model = Sequential(layers)

        x = Variable(
            [[1.0, 2.0, -1.0, 3.0], [0.0, -1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0]]
        )
        y = model.predict(x)

        expected = MatMul()(
            relu(MatMul()(x, layer1.params["W"]) + layer1.params["b"]),
            layer3.params["W"],
        )

        assert y.shape == (3, 2)
        assert allclose(y, expected)
