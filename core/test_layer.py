import numpy as np

from core.activation import sigmoid
from .layer import Layer


def test_predict():
    layer = Layer(2, 3)
    predict = layer.predict(np.array([1.0, 2.0]), sigmoid)
    assert predict.shape[0] == 3
