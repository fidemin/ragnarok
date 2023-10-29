import numpy as np
from matplotlib import pyplot as plt

from core.layer import Dropout
from core.net import NeuralNet
from core.optimizer import SGD
from language.layer import Embedding, GroupedLSTM, GroupedAffine, GroupedSoftmaxWithLoss
from language.util import process_text, WordIdConverter, convert_to_one_hot_encoding

if __name__ == '__main__':
    file_path = './data/100_word_sentences.txt'
    with open(file_path) as f:
        text = f.read().replace('\n', '<eos>').strip()

    # # the file is from: https://github.com/wojzaremba/lstm/blob/master/data/ptb.train.txt
    # file_path = '../downloads/ptb.train.txt'
    # with open(file_path) as f:
    #     text = f.read().replace('\n', '<eos>').strip()

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]
    xs = np.array(word_id_list[:-1])
    origin_ts = np.array(word_id_list[1:])
    ts = convert_to_one_hot_encoding(origin_ts, wi_converter.max_id())

    voca_size = wi_converter.number_of_words()
    wordvec_size = 100  # D
    hidden_size = 100  # H
    time_size = 10  # T

    original_data_size = xs.shape[0]
    mini_batch_size = 2
    max_epoch = 60
    learning_rate = 5.0
    iter_size = original_data_size // (mini_batch_size * time_size)

    # truncate data to data size for training
    data_size = iter_size * mini_batch_size * time_size
    xs = xs[:data_size]
    print("original data size: {}, truncated data size:, {}, voca_size: {}".format(original_data_size,
                                                                                   data_size, voca_size))

    W_emb = np.random.randn(voca_size, wordvec_size) * 0.01
    Wx_lstm1 = np.random.randn(wordvec_size, 4 * hidden_size) * (1 / np.sqrt(wordvec_size))
    Wh_lstm1 = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
    b_lstm1 = np.zeros((1, 4 * hidden_size))

    Wx_lstm2 = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
    Wh_lstm2 = np.random.randn(hidden_size, 4 * hidden_size) * (1 / np.sqrt(hidden_size))
    b_lstm2 = np.zeros((1, 4 * hidden_size))

    b_affine = np.zeros((1, voca_size))

    embedding_layer = Embedding(W_emb)
    lstm_layer1 = GroupedLSTM(Wx_lstm1, Wh_lstm1, b_lstm1, stateful=True)
    dropout_layer1 = Dropout(dropout_ratio=0.3)
    lstm_layer2 = GroupedLSTM(Wx_lstm2, Wh_lstm2, b_lstm2, stateful=True)
    dropout_layer2 = Dropout(dropout_ratio=0.3)
    affine_layer = GroupedAffine(W_emb.T, b_affine)  # share W with embedding layer
    loss_layer = GroupedSoftmaxWithLoss()

    net = NeuralNet([embedding_layer, lstm_layer1, dropout_layer1, lstm_layer2, dropout_layer2, affine_layer],
                    loss_layer, SGD(lr=1.0))
    ppl_list = []
    loss_list = []

    jump_size = data_size // mini_batch_size
    offsets_per_mini_batch = [i * jump_size for i in range(mini_batch_size)]

    for epoch in range(max_epoch):
        total_loss = 0
        loss_count = 0
        for iter_ in range(iter_size):
            mini_batch_X = np.empty((mini_batch_size, time_size), dtype=int)
            mini_batch_y = np.empty((mini_batch_size, time_size, voca_size), dtype=int)

            for t in range(time_size):
                for n, offset in enumerate(offsets_per_mini_batch):
                    position = offset + iter_ * time_size + t
                    # print("n: {}, position: {}".format(n, position))
                    mini_batch_X[n, t] = xs[position]
                    mini_batch_y[n, t, :] = ts[position]

            loss = net.forward(mini_batch_X, mini_batch_y)
            loss_list.append(loss)
            net.backward()
            net.optimize(grad_max_norm=0.25)

            if iter_ % 100 == 0:
                print('iter: {}, loss: {}'.format(iter_, loss))

            total_loss += loss
            loss_count += 1

        perplexity = np.exp(total_loss / loss_count)
        ppl_list.append(perplexity)
        print("epoch: {}, perplexity: {}".format(epoch + 1, perplexity))

    net.save_params('./data/100_word_params.pkl')

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
