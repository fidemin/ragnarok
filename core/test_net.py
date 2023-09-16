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
