from typing import List

import numpy as np
from keras.api.datasets import mnist

from ragnarok.core.config import using_backprop
from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.loss import SoftMaxLoss
from ragnarok.nn.layer.activation import ReLU
from ragnarok.nn.layer.linear import Linear
from ragnarok.nn.model.model import Model
from ragnarok.nn.optimizer.optimizer import Adam
from ragnarok.nn.train.util import accuracy
from ragnarok.utils.data.dataloader import DataLoader
from ragnarok.utils.data.dataset import Dataset


class MNISTModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(1000)
        self.relu = ReLU()
        self.fc2 = Linear(10)

    def predict(self, *tensors: Tensor, **kwargs) -> Tensor | List[Tensor]:
        x = tensors[0]
        h = self.fc1(x)
        h = self.relu(h)
        y = self.fc2(h)
        return y


class MNISTDataset(Dataset):
    def __init__(self, train: bool = True):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if train:
            self.data = x_train.reshape(-1, 784).astype("float32") / 255.0
            self.labels = np.eye(10)[y_train].astype("float32")
        else:
            self.data = x_test.reshape(-1, 784).astype("float32") / 255.0
            self.labels = np.eye(10)[y_test].astype("float32")

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    batch_size = 128
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochs = 20

    model = MNISTModel()
    loss_func = SoftMaxLoss()
    optimizer = Adam(lr=0.005)

    for epoch in range(epochs):
        train_sum_loss = 0
        train_sum_acc = 0

        for train_x, train_t in train_data_loader:
            x_batch = Tensor(train_x)
            t_batch = Tensor(train_t)

            y_batch = model.predict(x_batch)  # Changed from forward to predict
            loss = loss_func(y_batch, t_batch)

            acc = accuracy(y_batch, t_batch)

            model.zero_grad()
            loss.backward()

            train_sum_loss += loss.data * batch_size
            train_sum_acc += acc * batch_size

            optimizer.update(model.params)

        avg_train_loss = train_sum_loss / len(train_dataset)
        avg_train_acc = train_sum_acc / len(train_dataset)
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {avg_train_loss}, Accuracy: {avg_train_acc}")

        test_sum_loss = 0
        test_sum_acc = 0
        iter = 0
        with using_backprop(False):
            for test_x, test_t in test_data_loader:
                x_batch = Tensor(test_x)
                t_batch = Tensor(test_t)

                y_batch = model.predict(x_batch)  # Changed from forward to predict
                loss = loss_func(y_batch, t_batch)

                acc = accuracy(y_batch, t_batch)

                test_sum_loss += loss.data * batch_size
                test_sum_acc += acc * batch_size
                iter += 1

            avg_test_loss = test_sum_loss / len(test_dataset)
            avg_test_acc = test_sum_acc / len(test_dataset)
            print(f"Test Loss: {avg_test_loss}, Accuracy: {avg_test_acc}")
