from src.main.ragnarok.core.util import allclose
from src.main.ragnarok.core.variable import Variable
from src.main.ragnarok.nn.core.parameter import Parameter
from src.main.ragnarok.nn.optimizer.optimizer import SGD, Momentum


class TestSGD:
    def test_update(self):
        # Given
        param1 = Parameter([1.0, 2.0, 3.0])
        param2 = Parameter([4.0, 5.0, 6.0])
        param1.grad = Variable([1.0, 2.0, 3.0])
        param2.grad = Variable([4.0, 5.0, 6.0])

        # When
        optimizer = SGD(lr=0.01)
        optimizer.update([param1, param2])

        # Then
        assert allclose(param1, Parameter([0.99, 1.98, 2.97]))
        assert allclose(param2, Parameter([3.96, 4.95, 5.94]))


class TestMomentum:
    def test_update(self):
        # Given
        param1 = Parameter([1.0, 2.0, 3.0])
        param2 = Parameter([4.0, 5.0, 6.0])
        param1.grad = Variable([1.0, 2.0, 3.0])
        param2.grad = Variable([4.0, 5.0, 6.0])

        # When: First update
        optimizer = Momentum(lr=0.01, momentum=0.9)
        optimizer.update([param1, param2])

        # Then: First update
        # First update is same as SGD: velocity is 0 vector
        assert allclose(param1, Parameter([0.99, 1.98, 2.97]))
        assert allclose(param2, Parameter([3.96, 4.95, 5.94]))

        # When: Second update
        optimizer.update([param1, param2])

        # Then: Second update
        assert allclose(param1, Parameter([0.971, 1.942, 2.913]))
        assert allclose(param2, Parameter([3.884, 4.855, 5.826]))
