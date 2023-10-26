import numpy as np
from matplotlib import pyplot as plt

from core.net import NeuralNet
from core.optimizer import SGD
from language.layer import Embedding, GroupedLSTM, GroupedAffine, GroupedSoftmaxWithLoss
from language.util import process_text, WordIdConverter, convert_to_one_hot_encoding

if __name__ == '__main__':
    text = '''
        Importance of trade and commerce cannot be overstated; import and export activities drive the global economy. Countries import and export goods to meet their essential needs and promote economic growth. The import of raw materials is crucial for manufacturing, while the export of finished products can generate substantial revenue. Governments impose import tariffs and regulations to safeguard domestic industries, but these policies can impact international relations. In today's interconnected world, technology and information sharing are of utmost importance. The exchange of knowledge is as vital as physical imports and exports. As technology continues to advance, the importance of cybersecurity cannot be ignored, as it's necessary to protect against cyber threats. In a globalized world, understanding the intricate web of imports, exports, and information flows is key to thriving in the modern economy.
        '''

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]
    xs = np.array(word_id_list[:-1])
    origin_ts = np.array(word_id_list[1:])
    ts = convert_to_one_hot_encoding(origin_ts, wi_converter.max_id())

    voca_size = wi_converter.number_of_words()
    wordvec_size = 100  # D
    hidden_size = 100  # H
    time_subsequence_size = 10  # T

    data_size = word_id_list
    mini_batch_size = 1
    max_epoch = 100
    iter_size = voca_size // (mini_batch_size * time_subsequence_size)

    W_emb = np.random.randn(voca_size, wordvec_size) * 0.01
    Wx_lstm = np.random.randn(wordvec_size, 4 * hidden_size) * (1 / np.sqrt(wordvec_size))
    Wh_lstm = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
    b_lstm = np.zeros((1, 4 * hidden_size))

    W_affine = np.random.randn(hidden_size, voca_size) * (1 / np.sqrt(hidden_size))
    b_affine = np.zeros((1, voca_size))

    embedding_layer = Embedding(W_emb)
    lstm_layer = GroupedLSTM(Wx_lstm, Wh_lstm, b_lstm, stateful=True)
    affine_layer = GroupedAffine(W_affine, b_affine)
    loss_layer = GroupedSoftmaxWithLoss()

    net = NeuralNet([embedding_layer, lstm_layer, affine_layer], loss_layer, SGD(lr=5))
    ppl_list = []
    loss_list = []

    jump = voca_size // mini_batch_size

    for epoch in range(max_epoch):
        total_loss = 0
        loss_count = 0
        for iter_ in range(iter_size):
            mini_batch_X = np.empty((mini_batch_size, time_subsequence_size), dtype=int)
            mini_batch_y = np.empty((mini_batch_size, time_subsequence_size, voca_size), dtype=int)

            for t in range(time_subsequence_size):
                mini_batch_X[0, t] = xs[iter_ * time_subsequence_size + t]
                mini_batch_y[0, t, :] = ts[iter_ * time_subsequence_size + t]

            loss = net.forward(mini_batch_X, mini_batch_y)
            loss_list.append(loss)
            net.backward()
            net.optimize(grad_max_norm=0.25)

            total_loss += loss
            loss_count += 1

        perplexity = np.exp(total_loss / loss_count)
        ppl_list.append(perplexity)
        print("epoch: {}, perplexity: {}".format(epoch + 1, perplexity))

    plt.subplot(2, 1, 1)
    loss_x = list(range(1, len(loss_list) + 1))
    plt.plot(loss_x, loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    ppl_x = list(range(1, len(ppl_list) + 1))
    plt.plot(ppl_x, ppl_list)
    plt.xlabel('epoch')
    plt.ylabel('perplexity')

    plt.show()
