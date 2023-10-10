import numpy as np

from core.updater import SGD
from language.layer import CBOW


class TestCBOW:
    def test_forward(self):
        W_in = np.array([
            [0.01, 0.04],
            [0.02, 0.02],
            [0.05, 0.03]])
        x = np.array([
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0]]])

        expected = np.array([
            [(0.01 + 0.02) / 2.0, (0.04 + 0.02) / 2.0],
            [(0.05 + 0.02) / 2.0, (0.03 + 0.02) / 2.0],
            [(0.02 + 0.01) / 2.0, (0.02 + 0.04) / 2.0],
            [(0.05 + 0.02) / 2.0, (0.03 + 0.02) / 2.0]
        ])

        layer = CBOW(W_in, SGD())
        actual = layer.forward(x)
        np.allclose(actual, expected)

    def test_backward(self):
        W_in = np.array([
            [0.01, 0.04],
            [0.02, 0.02],
            [0.05, 0.03]])
        forward_input = np.array([
            [[1, 0, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0]]])

        backward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        layer = CBOW(W_in, SGD())
        layer.forward(forward_input)
        dx = layer.backward(backward_input)

        # dx[0], dx[1] should be same
        np.allclose(dx[0], np.dot(backward_input, W_in.T))
        np.allclose(dx[1], np.dot(backward_input, W_in.T))

        # check inner layers gradients
        inner_layer_1 = layer._sub_layers[0]
        grads1 = inner_layer_1.grads[0]
        np.allclose(grads1, np.dot(forward_input[:, 0].T, backward_input))

        inner_layer_2 = layer._sub_layers[1]
        grads2 = inner_layer_2.grads[0]
        np.allclose(grads2, np.dot(forward_input[:, 1].T, backward_input))

        np.allclose(layer.grads, inner_layer_1.grads[0] + inner_layer_2.grads[0])
