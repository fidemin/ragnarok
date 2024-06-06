import numpy as np

from src.main.core import activation
from src.main.core.layer import Dropout
from src.main.core import NeuralNet
from src.main.core import SGD
from language.layer import Embedding, GroupedLSTM, GroupedAffine, GroupedSoftmaxWithLoss
from language.util import process_text, WordIdConverter, convert_to_one_hot_encoding

if __name__ == '__main__':
    file_path = './data/100_word_sentences.txt'
    with open(file_path) as f:
        text = f.read().replace('\n', '<eos>').strip()

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]
    xs = np.array(word_id_list[:-1])
    origin_ts = np.array(word_id_list[1:])
    ts = convert_to_one_hot_encoding(origin_ts, wi_converter.max_id())

    voca_size = wi_converter.vocabulary_size()
    wordvec_size = 100  # D
    hidden_size = 100  # H
    time_size = 10  # T
    sample_size = 30

    original_data_size = xs.shape[0]
    print("original data size: {}, voca_size: {}".format(original_data_size, voca_size))

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
                    loss_layer, SGD(lr=1))
    net.load_params('./data/100_word_params.pkl')
    ppl_list = []
    loss_list = []

    start_id = wi_converter.word_to_id('do')
    x = start_id
    word_ids = [start_id]

    while len(word_ids) < sample_size:
        x = np.array(x).reshape(1, 1)
        word_prob = net.predict(x, train_flag=False)
        word_prob = activation.softmax(word_prob.flatten())
        sampled = np.random.choice(voca_size, size=1, p=word_prob)
        word_ids.append(sampled[0])
        x = sampled

    generated_words = [wi_converter.id_to_word(word_id) for word_id in word_ids]
    print(generated_words)
