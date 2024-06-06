import numpy as np

from src.main.core.layer import Affine, Sigmoid
from src.main.core.net import Net
from src.main.core.optimizer import SGD


def test_predict():
    W1 = np.array([[1.0, 2.0], [2.0, 3.0], [1.5, 1.0]])
    b1 = np.array([0.0, 0.0])
    W2 = np.array([[1.0, 2.0, 1.1], [0.0, 2.5, 4.5]])
    b2 = np.array([0.0, 0.0, 0.0])

    layer1 = Affine(W1, b1, SGD())
    layer2 = Sigmoid()
    layer3 = Affine(W2, b2, SGD())

    net = Net([layer1, layer2, layer3])
    x = np.array([[0.5, 0.7, 1.5], [1.1, 1.5, 0.2]])
    y = net.predict(x)

    assert y.shape == (2, 3)
