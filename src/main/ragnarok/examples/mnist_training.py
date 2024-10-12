import numpy as np
from keras.api.datasets import mnist

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.loss import SoftMaxLoss
from src.main.ragnarok.nn.layer.activation import Sigmoid
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Sequential
from src.main.ragnarok.nn.optimizer.optimizer import Adam

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    y_train = np.eye(10)[y_train].astype("float32")
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_test = np.eye(10)[y_test].astype("float32")

    epochs = 20
    batch_size = 100
    iter_num = len(x_train) // batch_size

    model = Sequential([Linear(1000), Sigmoid(), Linear(10)])
    loss_func = SoftMaxLoss()
    optimizer = Adam(lr=0.01)

    for epoch in range(epochs):
        for i in range(iter_num):
            x_batch = Variable(x_train[batch_size * i : batch_size * (i + 1)])
            y_batch = Variable(y_train[batch_size * i : batch_size * (i + 1)])

            model.zero_grad()
            model.predict(x_batch)

            y = model.predict(x_batch)  # Changed from forward to predict
            loss = loss_func(y, y_batch)

            loss.backward()

            optimizer.update(model.params)

        print(f"Epoch: {epoch}, Loss: {loss.data}")
