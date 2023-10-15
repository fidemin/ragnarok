import numpy as np

from core.updater import SGD
from language.layer import CBOW, MultiInputEmbedding


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
        assert np.allclose(actual, expected)

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

        expected_dW = np.array([
            [(0.1 + 0.4), (0.4 + 0.7)],
            [(0.1 + 0.2 + 0.4 + 0.3), (0.4 + 0.5 + 0.7 + 0.4)],
            [(0.2 + 0.3), (0.5 + 0.4)]
        ]) / 2.0

        layer = CBOW(W_in, SGD())
        layer.forward(forward_input)
        dx = layer.backward(backward_input)

        # dW
        assert np.allclose(layer.grads[0], expected_dW)

        # dx[0], dx[1] should be same
        assert np.allclose(dx[0], np.dot(backward_input, W_in.T))
        assert np.allclose(dx[1], np.dot(backward_input, W_in.T))

        # check inner layers gradients
        inner_layer_1 = layer._sub_layers[0]
        grads1 = inner_layer_1.grads[0]
        assert np.allclose(grads1, np.dot(forward_input[:, 0].T, backward_input))

        inner_layer_2 = layer._sub_layers[1]
        grads2 = inner_layer_2.grads[0]
        assert np.allclose(grads2, np.dot(forward_input[:, 1].T, backward_input))

        assert np.allclose(layer.grads, inner_layer_1.grads[0] + inner_layer_2.grads[0])


class TestMultiInputEmbedding:
    def test_forward(self):
        W_in = np.array([
            [0.01, 0.04],
            [0.02, 0.02],
            [0.05, 0.03]])

        x = np.array([
            [0, 1],
            [2, 1],
            [1, 0],
            [2, 1]])

        expected = np.array([
            [(0.01 + 0.02) / 2.0, (0.04 + 0.02) / 2.0],
            [(0.05 + 0.02) / 2.0, (0.03 + 0.02) / 2.0],
            [(0.02 + 0.01) / 2.0, (0.02 + 0.04) / 2.0],
            [(0.05 + 0.02) / 2.0, (0.03 + 0.02) / 2.0]
        ])

        layer = MultiInputEmbedding(W_in, SGD())
        actual = layer.forward(x)
        assert np.allclose(actual, expected)

    def test_backward(self):
        W_in = np.array([
            [0.01, 0.04],
            [0.02, 0.02],
            [0.05, 0.03]])

        forward_input = np.array([
            [0, 1],
            [2, 1],
            [1, 0],
            [2, 1]])

        backward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        expected_dW = np.array([
            [(0.1 + 0.4), (0.4 + 0.7)],
            [(0.1 + 0.2 + 0.4 + 0.3), (0.4 + 0.5 + 0.7 + 0.4)],
            [(0.2 + 0.3), (0.5 + 0.4)]
        ]) / 2.0

        layer = MultiInputEmbedding(W_in, SGD())
        layer.forward(forward_input)
        dx = layer.backward(backward_input)

        # dW calculation check
        assert np.allclose(layer.grads[0], expected_dW)

        # dx[0] from indexes [0, 2, 1, 2], dx[1] from indexes [1, 1, 0, 1]
        assert np.allclose(dx[0], (backward_input * W_in[forward_input[:, 0]]).sum(axis=1))
        assert np.allclose(dx[1], (backward_input * W_in[forward_input[:, 1]]).sum(axis=1))

    def test_update_params(self):
        W_in = np.array([
            [0.01, 0.04],
            [0.02, 0.02],
            [0.05, 0.03]])

        forward_input = np.array([
            [0, 1],
            [2, 1],
            [1, 0],
            [2, 1]])

        backward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        expected_dW = np.array([
            [(0.1 + 0.4), (0.4 + 0.7)],
            [(0.1 + 0.2 + 0.4 + 0.3), (0.4 + 0.5 + 0.7 + 0.4)],
            [(0.2 + 0.3), (0.5 + 0.4)]
        ]) / 2.0

        lr = 0.01

        # SGD calculation
        expected_W = W_in - lr * expected_dW

        layer = MultiInputEmbedding(W_in, SGD(lr=lr))

        layer.forward(forward_input)
        layer.backward(backward_input)

        layer.update_params()

        actual_W = layer.params[0]
        assert np.allclose(actual_W, expected_W)

        # W of sub layer also should be updated
        for sub_layer in layer._sub_layers:
            assert np.allclose(sub_layer.params[0], actual_W)

