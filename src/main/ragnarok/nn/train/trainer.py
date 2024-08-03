from src.main.ragnarok.core.function import Function
from src.main.ragnarok.nn.model.model import Model
from src.main.ragnarok.nn.optimizer.optimizer import Optimizer


class Trainer:
    def __init__(
        self,
        *,
        model: Model,
        loss_func: Function,
        optimizer: Optimizer,
        epochs: int,
        verbose=True,
        print_interval=1000,
    ):
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.print_interval = print_interval

    def train(self, x, t):
        for i in range(self.epochs):
            epoch = i + 1
            for param in self.model.params:
                param.clear_grad()

            y = self.model.predict(x)  # Changed from forward to predict
            loss = self.loss_func(y, t)

            loss.backward()

            self.optimizer.update(self.model.params)

            if self.verbose and i % self.print_interval == 0:
                print(f"Epoch: {epoch}, Loss: {loss.data}")
