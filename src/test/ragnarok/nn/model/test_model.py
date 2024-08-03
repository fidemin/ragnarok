from typing import List

from src.main.ragnarok.core.function import MatMul
from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.activation import relu
from src.main.ragnarok.nn.layer.activation import ReLU
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Sequential, Model


class MockModel(Model):
    def __init__(self):
        super().__init__()
        self.layer_linear_1 = Linear(8)
        self.layer_relu = ReLU()
        self.layer_linear_2 = Linear(4, 8, use_bias=False)

    def predict(self, *variables: Variable, **kwargs) -> Variable | List[Variable]:
        x = variables[0]
        h = self.layer_linear_1.forward(x)
        h = self.layer_relu.forward(h)
        y = self.layer_linear_2.forward(h)
        return y


class TestModel:
    def test_init(self):
        model = MockModel()

        assert list(model.params) == [
            model.layer_linear_2.params["W"],
        ]

        x = Variable([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])

        # to initialize first linear layer's parameter: Linear layer's param is lazily loaded without input size
        model.predict(x)

        assert list(model.params) == [
            model.layer_linear_1.params["W"],
            model.layer_linear_1.params["b"],
            model.layer_linear_2.params["W"],
        ]


class TestSequential:
    def test_init(self):
        layer1 = Linear(5, 4)
        layer2 = ReLU()
        layer3 = Linear(2, 5, use_bias=False, name="Last")
        layers = [layer1, layer2, layer3]
        model = Sequential(layers)

        assert model.layers_dict == {
            "Linear_1": layer1,
            "ReLU_2": layer2,
            "Last_3": layer3,
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
