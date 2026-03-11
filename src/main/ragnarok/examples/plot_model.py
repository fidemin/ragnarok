from typing import List

import numpy as np
from keras.api.datasets import mnist

from src.main.ragnarok.core.tensor import Tensor
from src.main.ragnarok.graph.plot import plot_tensor_graph
from src.main.ragnarok.nn.layer.activation import Sigmoid
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Model, MLP


class MNISTModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1000)
        self.sigmoid = Sigmoid()
        self.fc2 = Linear(10)

    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        h = self.fc1(x)
        h = self.sigmoid(h)
        y = self.fc2(h)
        return y


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    y_train = np.eye(10)[y_train].astype("float32")
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    y_test = np.eye(10)[y_test].astype("float32")

    model = MLP(out_sizes=[1000, 10], activation="sigmoid")
    y = model.predict(Tensor(x_train[:1]))

    plot_tensor_graph(y, output_file="temp/mlp_model_graph.png", temp_dir="temp")
