from typing import List

from ragnarok.core.function.math import MatMul
from ragnarok.core.tensor import Tensor
from ragnarok.core.util import allclose
from ragnarok.nn.function.activation import relu
from ragnarok.nn.layer.activation import ReLU
from ragnarok.nn.layer.linear import Linear
from ragnarok.nn.model.model import Sequential, Model, MLP


class MockModel(Model):
    def __init__(self):
        super().__init__()
        self.layer_linear_1 = Linear(8)
        self.layer_relu = ReLU()
        self.layer_linear_2 = Linear(4, 8, use_bias=False)

    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
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

        x = Tensor([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])

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

        x = Tensor([[1.0, 2.0, -1.0, 3.0], [0.0, -1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0]])
        y = model.predict(x)

        expected = MatMul()(
            relu(MatMul()(x, layer1.params["W"]) + layer1.params["b"]),
            layer3.params["W"],
        )

        assert y.shape == (3, 2)
        assert allclose(y, expected)


class TestMLP:
    def test_predict(self):
        model = MLP(out_sizes=(8, 4, 2), activation="relu")

        x = Tensor([[1.0, 2.0, -1.0, 3.0], [0.0, -1.0, 2.0, 1.0], [2.0, 0.0, 1.0, 2.0]])
        y = model.predict(x)

        expected = (
            MatMul()(
                relu(
                    MatMul()(
                        relu(
                            MatMul()(x, model.layers_dict["linear_1"].params["W"])
                            + model.layers_dict["linear_1"].params["b"]
                        ),
                        model.layers_dict["linear_2"].params["W"],
                    )
                    + model.layers_dict["linear_2"].params["b"]
                ),
                model.layers_dict["linear_3"].params["W"],
            )
            + model.layers_dict["linear_3"].params["b"]
        )

        assert y.shape == (3, 2)
        assert allclose(y, expected)
