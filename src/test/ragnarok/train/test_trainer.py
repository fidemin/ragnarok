import numpy as np

from src.main.ragnarok.core.function import Function
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter
from src.main.ragnarok.nn.function.loss import MeanSquaredError
from src.main.ragnarok.nn.layer.activation import ReLU
from src.main.ragnarok.nn.layer.linear import Linear
from src.main.ragnarok.nn.model.model import Model, Sequential
from src.main.ragnarok.nn.optimizer.optimizer import Optimizer, Adam
from src.main.ragnarok.nn.train.trainer import Trainer


class TestTrainer:
    def test_train_with_mock(self, mocker):
        # Given
        mock_model = mocker.Mock(spec=Model)
        num_of_params = 10
        mock_model.params = [mocker.Mock(spec=Parameter) for _ in range(num_of_params)]

        mock_model.predict.return_value = mocker.Mock(spec=Variable)

        mock_loss_func = mocker.Mock(spec=Function)
        mock_loss_func.return_value = mocker.Mock(spec=Variable)

        mock_optimizer = mocker.Mock(spec=Optimizer)

        x = mocker.Mock(spec=Variable)
        t = mocker.Mock(spec=Variable)

        epochs = 100

        # When
        trainer = Trainer(
            model=mock_model,
            loss_func=mock_loss_func,
            optimizer=mock_optimizer,
            epochs=epochs,
            verbose=True,
            print_interval=10,
        )
        trainer.train(x, t)

        # Then
        assert mock_model.predict.call_count == epochs
        assert mock_loss_func.call_count == epochs
        assert mock_loss_func.return_value.backward.call_count == epochs
        assert mock_optimizer.update.call_count == epochs
        assert mock_optimizer.update.called_with(mock_model.params)

    def test_train_runnable(self):
        layer1 = Linear(8)
        layer2 = ReLU()
        layer3 = Linear(4, 8, use_bias=False)

        model = Sequential([layer1, layer2, layer3])

        loss_func = MeanSquaredError()
        optimizer = Adam(lr=0.01)

        x = Variable(np.random.randn(10, 8))
        t = Variable(np.random.randn(10, 4))

        trainer = Trainer(
            model=model, optimizer=optimizer, loss_func=loss_func, epochs=1000
        )

        trainer.train(x, t)
