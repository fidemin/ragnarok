import numpy as np
import pytest

from .activation import sigmoid
from .net import Net, NetInitException


@pytest.mark.parametrize(
    "test_input,num_layers,weight_shapes",
    [([2, 3], 1, [(2, 3)]), ([2, 3, 10], 2, [(2, 3), (3, 10)])]
)
def test_net(test_input, num_layers, weight_shapes):
    net = Net(test_input)
    assert net.activation_func == sigmoid
    assert len(net.layers) == num_layers

    for i in range(len(net.layers)):
        assert net.layers[i].W.shape == weight_shapes[i]


@pytest.mark.parametrize(
    "test_input",
    [([]), ([2])]
)
def test_net_exception(test_input):
    with pytest.raises(NetInitException):
        Net([])


@pytest.mark.parametrize(
    "test_input,output_shape",
    [([2, 3], (3,)), ([2, 3, 10], (10,))]
)
def test_predict(test_input, output_shape):
    init_x = np.array([3.0, 1.0])
    net = Net(test_input)
    actual = net.predict(init_x)
    assert actual.shape == output_shape
    assert np.allclose(abs(np.sum(actual)), 1)


@pytest.mark.parametrize(
    "net_size,t",
    [([2, 3], np.array([0, 1, 0])), ([2, 3, 10], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))]
)
def test_loss(net_size, t):
    init_x = np.array([3.0, 1.0])
    net = Net(net_size)
    actual = net.loss(init_x, t)
    assert actual >= 0.0


@pytest.mark.parametrize(
    "net_size,t",
    [([2, 3], np.array([0, 1, 0])), ([2, 3, 10], np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))]
)
def test_gradient(net_size, t):
    init_x = np.array([[3.0, 1.0], [5.0, 2.0]])
    net = Net(net_size)
    net.gradient(init_x, t)

    for i in range(len(net_size) - 1):
        layer = net.layers[i]
        assert layer.W_grad.shape == (net_size[i], net_size[i + 1])
        assert np.all(layer.W_grad != 0)
        assert layer.b_grad.shape == (net_size[i + 1],)
        assert np.all(layer.b_grad != 0)
