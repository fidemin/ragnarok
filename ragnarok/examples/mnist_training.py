from typing import List

import numpy as np
from keras.api.datasets import mnist

from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.loss import SoftMaxLoss
from ragnarok.nn.layer.activation import Sigmoid
from ragnarok.nn.layer.linear import Linear
from ragnarok.nn.model.model import Model
from ragnarok.nn.optimizer.optimizer import Adam


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

    epochs = 20
    batch_size = 100
    iter_num = len(x_train) // batch_size

    model = MNISTModel()
    loss_func = SoftMaxLoss()
    optimizer = Adam(lr=0.01)

    for epoch in range(epochs):
        for i in range(iter_num):
            x_batch = Tensor(x_train[batch_size * i : batch_size * (i + 1)])
            t_batch = Tensor(y_train[batch_size * i : batch_size * (i + 1)])

            model.zero_grad()
            model.predict(x_batch)

            y_batch = model.predict(x_batch)  # Changed from forward to predict
            loss = loss_func(y_batch, t_batch)

            loss.backward()

            optimizer.update(model.params)

        print(f"Epoch: {epoch}, Loss: {loss.data}")
