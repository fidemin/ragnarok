import numpy as np

from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.function.loss import MeanSquaredError
from src.main.ragnarok.nn.layer.activation import ReLU
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Sequential
from src.main.ragnarok.nn.optimizer.optimizer import Adam
from src.main.ragnarok.nn.train.trainer import Trainer

if __name__ == "__main__":
    layer1 = Linear(8)
    layer2 = ReLU()
    layer3 = Linear(4, 8, use_bias=False)

    model = Sequential([layer1, layer2, layer3])

    loss_func = MeanSquaredError()
    optimizer = Adam(lr=0.01)

    x = Variable(np.random.randn(10, 8))
    t = Variable(np.random.randn(10, 4))

    trainer = Trainer(
        model=model, optimizer=optimizer, loss_func=loss_func, epochs=10000
    )

    trainer.train(x, t)
