import copy

import numpy as np

from core.updater import SGD
from language.layer import CBOW, CBOWInputEmbedding, EmbeddingDot


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


class TestEmbedingDot:
    def test_forward(self):
        # m * h
        forward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        # m * 1
        indexes = [1, 4, 3, 0]

        # h * out
        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01],
            [0.02, 0.02, 0.07, 0.01, 0.04],
        ])

        expected_out = np.array([
            [0.04 * 0.1 + 0.02 * 0.4],
            [0.01 * 0.2 + 0.04 * 0.5],
            [0.04 * 0.4 + 0.01 * 0.7],
            [0.01 * 0.3 + 0.02 * 0.4]
        ])

        layer = EmbeddingDot(W, SGD())
        kwargs = {EmbeddingDot.indexes_key: indexes}
        out = layer.forward(forward_input, **kwargs)
        assert np.allclose(out, expected_out)

    def test_backward(self):
        # m * h
        forward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        # m * 1
        indexes = [1, 4, 3, 1]

        # h * out
        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01],
            [0.02, 0.02, 0.07, 0.01, 0.04],
        ])

        dout = np.array([
            [0.4], [0.7], [0.5], [0.1]
        ])

        expected_dx = np.array([
            [W[0, indexes[0]] * dout[0][0], W[1, indexes[0]] * dout[0][0]],
            [W[0, indexes[1]] * dout[1][0], W[1, indexes[1]] * dout[1][0]],
            [W[0, indexes[2]] * dout[2][0], W[1, indexes[2]] * dout[2][0]],
            [W[0, indexes[3]] * dout[3][0], W[1, indexes[3]] * dout[3][0]],
        ])

        expected_dW = np.array([
            [0.0,
             forward_input[0][0] * dout[0][0] + forward_input[3][0] * dout[3][0],
             0.0,
             forward_input[2][0] * dout[2][0],
             forward_input[1][0] * dout[1][0]],
            [0.0,
             forward_input[0][1] * dout[0][0] +
             forward_input[3][1] * dout[3][0],
             0.0,
             forward_input[2][1] * dout[2][0],
             forward_input[1][1] * dout[1][0]]
        ])

        layer = EmbeddingDot(W, SGD())
        kwargs = {EmbeddingDot.indexes_key: indexes}
        layer.forward(forward_input, **kwargs)
        actual_dx = layer.backward(dout)

        actual_dW = layer.grads[0]
        assert np.allclose(actual_dx, expected_dx)
        assert np.allclose(actual_dW, expected_dW)

    def test_update_params(self):
        # m * h
        forward_input = np.array([
            [0.1, 0.4],
            [0.2, 0.5],
            [0.4, 0.7],
            [0.3, 0.4]
        ])

        # m * 1
        indexes = [1, 4, 3, 1]

        # h * out
        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01],
            [0.02, 0.02, 0.07, 0.01, 0.04],
        ])

        W_original = copy.deepcopy(W)

        dout = np.array([
            [0.4], [0.7], [0.5], [0.1]
        ])

        lr = 0.01
        layer = EmbeddingDot(W, SGD(lr=lr))
        kwargs = {EmbeddingDot.indexes_key: indexes}
        layer.forward(forward_input, **kwargs)
        layer.backward(dout)
        dW = layer.grads[0]

        layer.update_params()

        expected_W = W_original - lr * dW
        assert np.allclose(layer.params[0], expected_W)
        assert np.allclose(layer._embedding_layer.params[0], expected_W.T)


class TestCBOWInputEmbedding:
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

        layer = CBOWInputEmbedding(W_in, SGD())
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

        layer = CBOWInputEmbedding(W_in, SGD())
        layer.forward(forward_input)
        dx = layer.backward(backward_input)

        # dW calculation check
        assert np.allclose(layer.grads[0], expected_dW)

        # dx[0] from indexes [0, 2, 1, 2], dx[1] from indexes [1, 1, 0, 1]
        assert dx is None

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

        layer = CBOWInputEmbedding(W_in, SGD(lr=lr))

        layer.forward(forward_input)
        layer.backward(backward_input)

        layer.update_params()

        actual_W = layer.params[0]
        assert np.allclose(actual_W, expected_W)

        # W of sub layer also should be updated
        for sub_layer in layer._sub_layers:
            assert np.allclose(sub_layer.params[0], actual_W)
