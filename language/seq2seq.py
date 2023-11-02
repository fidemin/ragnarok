import numpy as np

from core.layer import Layer
from language.layer import Embedding, GroupedLSTM, GroupedAffine


class Encoder:
    def __init__(self, voca_size, wordvec_size, hidden_size):
        W_emb = np.random.randn(voca_size, wordvec_size) * 0.01
        Wx_lstm = np.random.randn(wordvec_size, 4 * hidden_size) * (1 / np.sqrt(wordvec_size))
        Wh_lstm = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
        b_lstm = np.zeros((1, 4 * hidden_size))

        embedding_layer = Embedding(W_emb)
        lstm_layer = GroupedLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=False)

        self._embedding_layer = embedding_layer
        self._lstm_layer = lstm_layer
        self._hs = None

        self.params = self._embedding_layer.params + self._lstm_layer.params
        self.grads = self._embedding_layer.grads + self._lstm_layer.grads

    def forward(self, xs: np.ndarray):
        ds = self._embedding_layer.forward(xs)
        hs = self._lstm_layer.forward(ds)
        # return only last sequence of hs
        self._hs = hs
        return hs[:, -1, :]

    def backward(self, dh: np.ndarray):
        dhs = np.zeros_like(self._hs)
        dhs[:, -1, :] = dh
        dout = self._lstm_layer.backward(dhs)
        self._embedding_layer.backward(dout)


class Decoder:
    def __init__(self, voca_size, wordvec_size, hidden_size):
        W_emb = np.random.randn(voca_size, wordvec_size) * 0.01
        Wx_lstm = np.random.randn(wordvec_size, 4 * hidden_size) * (1 / np.sqrt(wordvec_size))
        Wh_lstm = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
        b_lstm = np.zeros((1, 4 * hidden_size))
        W_affine = np.random.randn(hidden_size, voca_size) * (1 / np.sqrt(hidden_size))
        b_affine = np.zeros((1, voca_size))

        embedding_layer = Embedding(W_emb)
        lstm_layer = GroupedLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=True)
        affine_layer = GroupedAffine(W_affine, b_affine)

        self._embedding_layer = embedding_layer
        self._lstm_layer = lstm_layer
        self._affine_layer = affine_layer

        self.params = self._embedding_layer.params + self._lstm_layer.params + self._affine_layer.params
        self.grads = self._embedding_layer.grads + self._lstm_layer.grads + self._affine_layer.grads

    def forward(self, xs: np.ndarray, h: np.ndarray):
        # TODO: The way to insert to inner variable is required for Layer
        self._set_lstm_state(h)
        out = self._embedding_layer.forward(xs)
        out = self._lstm_layer.forward(out)
        out = self._affine_layer.forward(out)
        return out

    def _set_lstm_state(self, h: np.ndarray):
        # c is need to reset for every forward
        self._lstm_layer._h = h
        self._lstm_layer._c = None

    def backward(self, dout: np.ndarray):
        dout = self._affine_layer.backward(dout)
        dout = self._lstm_layer.backward(dout)
        self._embedding_layer.backward(dout)

        # TODO: The way to insert to get variable is required for Layer
        return self._lstm_layer._dh

    def generate(self, start_id, h, sample_size):
        self._set_lstm_state(h)

        sample_word_id = start_id

        word_id_list = []

        for _ in range(sample_size):
            xs = np.array(sample_word_id).reshape((1, 1))
            out = self._embedding_layer.forward(xs)
            out = self._lstm_layer.forward(out)
            out = self._affine_layer.forward(out)

            sample_word_id = np.argmax(out.flatten())
            word_id_list.append(int(sample_word_id))

        return word_id_list


class PeekyDecoder(Decoder):
    def __init__(self, voca_size, wordvec_size, hidden_size):
        W_emb = np.random.randn(voca_size, wordvec_size) * 0.01
        Wx_lstm = np.random.randn(wordvec_size + hidden_size, 4 * hidden_size) * (1 / np.sqrt(wordvec_size))
        Wh_lstm = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
        b_lstm = np.zeros((1, 4 * hidden_size))
        W_affine = np.random.randn(hidden_size + hidden_size, voca_size) * (1 / np.sqrt(hidden_size))
        b_affine = np.zeros((1, voca_size))

        embedding_layer = Embedding(W_emb)
        lstm_layer = GroupedLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=True)
        affine_layer = GroupedAffine(W_affine, b_affine)

        self._embedding_layer = embedding_layer
        self._lstm_layer = lstm_layer
        self._affine_layer = affine_layer
        self._hidden_size = hidden_size

        self.params = self._embedding_layer.params + self._lstm_layer.params + self._affine_layer.params
        self.grads = self._embedding_layer.grads + self._lstm_layer.grads + self._affine_layer.grads

    def forward(self, xs: np.ndarray, h: np.ndarray):
        N, T = xs.shape
        N, H = h.shape

        # TODO: The way to insert to inner variable is required for Layer
        self._set_lstm_state(h)
        out = self._embedding_layer.forward(xs)

        hs = np.repeat(h, T, axis=0).reshape((N, T, H))
        out = np.concatenate((hs, out), axis=2)
        out = self._lstm_layer.forward(out)

        out = np.concatenate((hs, out), axis=2)
        out = self._affine_layer.forward(out)
        return out

    def backward(self, dout: np.ndarray):
        dout = self._affine_layer.backward(dout)
        dh1, dout = np.split(dout, [self._hidden_size], axis=2)
        dh1 = dh1.sum(axis=1)
        dout = self._lstm_layer.backward(dout)
        dh2, dout = np.split(dout, [self._hidden_size], axis=2)
        dh2 = dh2.sum(axis=1)
        self._embedding_layer.backward(dout)

        # TODO: The way to insert to get variable is required for Layer
        return self._lstm_layer._dh + dh1 + dh2

    def generate(self, start_id, h, sample_size):
        self._set_lstm_state(h)

        sample_word_id = start_id

        word_id_list = []

        for _ in range(sample_size):
            xs = np.array(sample_word_id).reshape((1, 1))
            out = self._embedding_layer.forward(xs)

            N, T = xs.shape
            hs = np.repeat(h, T, axis=0).reshape((N, T, self._hidden_size))
            out = np.concatenate((hs, out), axis=2)
            out = self._lstm_layer.forward(out)
            out = np.concatenate((hs, out), axis=2)
            out = self._affine_layer.forward(out)

            sample_word_id = np.argmax(out.flatten())
            word_id_list.append(int(sample_word_id))

        return word_id_list


class Seq2Seq(Layer):
    decoder_xs_key = 'decoder_xs'

    def __init__(self, voca_size, wordvec_size, hidden_size):
        encoder = Encoder(voca_size, wordvec_size, hidden_size)
        decoder = Decoder(voca_size, wordvec_size, hidden_size)

        self._encoder = encoder
        self._decoder = decoder

        self.params = self._encoder.params + self._decoder.params
        self.grads = self._encoder.grads + self._decoder.grads

    def forward(self, xs: np.ndarray, **kwargs):
        decoder_xs = kwargs[self.decoder_xs_key]
        h = self._encoder.forward(xs)
        out = self._decoder.forward(decoder_xs, h)
        return out

    def backward(self, dout: np.ndarray):
        dh = self._decoder.backward(dout)
        self._encoder.backward(dh)

    def update_params(self):
        pass

    def generate(self, xs: np.ndarray, start_id, sample_size):
        h = self._encoder.forward(xs)
        return self._decoder.generate(start_id, h, sample_size)


class PeekySeq2Seq(Seq2Seq):
    decoder_xs_key = 'decoder_xs'

    def __init__(self, voca_size, wordvec_size, hidden_size):
        encoder = Encoder(voca_size, wordvec_size, hidden_size)
        decoder = PeekyDecoder(voca_size, wordvec_size, hidden_size)

        self._encoder = encoder
        self._decoder = decoder

        self.params = self._encoder.params + self._decoder.params
        self.grads = self._encoder.grads + self._decoder.grads
