import copy
from unittest import mock

import numpy as np

from core.updater import SGD
from language.layer import CBOWInput, CBOWInputEmbedding, EmbeddingDot, NegativeSampling, LSTM
from language.util import UnigramSampler


class TestCBOWInput:
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

        layer = CBOWInput(W_in, SGD())
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

        layer = CBOWInput(W_in, SGD())
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


class TestNegativeSampling:

    def test_forward(self):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 4, 3]
        UnigramSampler(word_id_list)
        sampler_class = mock.Mock(spec=UnigramSampler)
        sampler_instance = sampler_class.return_value
        sampler_instance.sample.return_value = np.array([[2, 3], [1, 4]])

        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01, 0.04],
            [0.02, 0.02, 0.07, 0.01, 0.04, 0.02],
            [0.12, 0.31, 0.03, 0.09, 0.04, 0.22],
        ])

        forward_input = np.array([
            [0.2, 0.3, 0.5],
            [0.6, 0.2, 0.7]
        ])

        positive_indexes = np.array([1, 5])

        negative_size = 2
        layer = NegativeSampling(W, negative_size, sampler_instance, SGD())
        kwargs = {NegativeSampling.positive_indexes_key: positive_indexes}

        layer.forward(forward_input, **kwargs)

    def test_backward(self):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 4, 3]
        UnigramSampler(word_id_list)
        sampler_class = mock.Mock(spec=UnigramSampler)
        sampler_instance = sampler_class.return_value
        sampler_instance.sample.return_value = np.array([[2, 3], [1, 4]])

        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01, 0.04],
            [0.02, 0.02, 0.07, 0.01, 0.04, 0.02],
            [0.12, 0.31, 0.03, 0.09, 0.04, 0.22],
        ])

        forward_input = np.array([
            [0.2, 0.3, 0.5],
            [0.6, 0.2, 0.7]
        ])

        positive_indexes = np.array([1, 5])

        negative_size = 2
        layer = NegativeSampling(W, negative_size, sampler_instance, SGD())
        kwargs = {NegativeSampling.positive_indexes_key: positive_indexes}

        layer.forward(forward_input, **kwargs)
        dout = 1
        layer.backward(dout)

        dW_sub_sum = np.zeros(layer.grads[0].shape)

        for sub_l in layer._embedding_dot_layers:
            dW_sub_sum += sub_l.grads[0]

        assert np.allclose(layer.grads[0], dW_sub_sum)

    def test_update_params(self):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 4, 3]
        UnigramSampler(word_id_list)
        sampler_class = mock.Mock(spec=UnigramSampler)
        sampler_instance = sampler_class.return_value
        sampler_instance.sample.return_value = np.array([[2, 3], [1, 4]])

        W = np.array([
            [0.01, 0.04, 0.02, 0.04, 0.01, 0.04],
            [0.02, 0.02, 0.07, 0.01, 0.04, 0.02],
            [0.12, 0.31, 0.03, 0.09, 0.04, 0.22],
        ])

        W_original = copy.deepcopy(W)

        forward_input = np.array([
            [0.2, 0.3, 0.5],
            [0.6, 0.2, 0.7]
        ])

        positive_indexes = np.array([1, 5])

        negative_size = 2
        layer = NegativeSampling(W, negative_size, sampler_instance, SGD())
        kwargs = {NegativeSampling.positive_indexes_key: positive_indexes}

        layer.forward(forward_input, **kwargs)
        dout = 1
        layer.backward(dout)
        assert np.allclose(W_original, layer.params[0])

        layer.update_params()

        W_new = layer.params[0]
        assert not np.allclose(W_original, W_new)

        for embedding_dot_layer in layer._embedding_dot_layers:
            assert np.allclose(W_new, embedding_dot_layer.params[0])
            assert np.allclose(W_new, embedding_dot_layer._embedding_layer.params[0].T)


class TestLSTM:
    def test_forward(self):
        Wx = np.array([
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8],
        ])

        Wh = np.array([
            [-0.1, -0.2, 0.2, 0.1, 0.3, 0.6, 0.7, 0.8],
            [0.5, 1.2, 1.2, 1.7, 1.1, 1.6, 2.7, 2.0],
        ])

        b = np.array([
            [0.4, 0.5, 0.1, 0.2, 0.3, 0.7, 0.8, 0.6]
        ])

        x = np.array([
            [0.3, 0.2, 0.7]
        ])
        h_prev = np.array([
            [1.1, 2.4]
        ])
        c_prev = np.array([
            [1.1, 2.5]
        ])

        lstm = LSTM(Wx, Wh, b, SGD())

        kwargs = {
            LSTM.h_prev_key: h_prev,
            LSTM.c_prev_key: c_prev
        }

        lstm.forward(x, **kwargs)
