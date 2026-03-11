import numpy as np

from ragnarok.core.tensor import Tensor
from ragnarok.nn.function.loss import MeanSquaredError
from ragnarok.nn.layer.activation import ReLU
from ragnarok.nn.layer.linear import Linear
from ragnarok.nn.model.model import Sequential
from ragnarok.nn.optimizer.optimizer import Adam
from ragnarok.nn.train.trainer import Trainer

if __name__ == "__main__":
    layer1 = Linear(8)
    layer2 = ReLU()
    layer3 = Linear(4, 8, use_bias=False)

    model = Sequential([layer1, layer2, layer3])

    loss_func = MeanSquaredError()
    optimizer = Adam(lr=0.001)

    x = Tensor(np.random.randn(10, 8))
    t = Tensor(np.random.randn(10, 4))

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_func=loss_func, epochs=10000
    )

    trainer.train(x, t)
