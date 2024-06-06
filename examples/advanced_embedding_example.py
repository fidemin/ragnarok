import time

import numpy as np
from matplotlib import pyplot as plt

from src.main.core import Net
from src.main.core import Adam
from language.layer import CBOWInputEmbedding, NegativeSampling
from language.util import process_text, ContextTargetConverter, WordIdConverter, UnigramSampler, most_similar_words

if __name__ == '__main__':
    # the file is from: https://github.com/wojzaremba/lstm/blob/master/data/ptb.train.txt
    file_path = '../downloads/ptb.train.txt'
    with open(file_path) as f:
        text = f.read().replace('\n', '<eos>').strip()

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]

    # window_size = 5
    window_size = 2
    context_target_converter = ContextTargetConverter(word_id_list, window_size=window_size)
    contexts = context_target_converter.contexts()
    target = context_target_converter.targets()

    # build network
    input_layer_size = wi_converter.max_id() + 1
    hidden_layer_size = 100

    init_weight1 = 1 / np.sqrt(input_layer_size)
    # init_weight1 = 0.01
    layer1 = CBOWInputEmbedding.from_size(input_layer_size, hidden_layer_size, Adam(), init_weight=init_weight1)
    sampler = UnigramSampler(word_id_list)
    negative_size = 10
    init_weight2 = 1 / np.sqrt(hidden_layer_size)
    # init_weight2 = 0.01
    layer2 = NegativeSampling.from_size(hidden_layer_size, input_layer_size, negative_size, sampler, Adam(),
                                        init_weight=init_weight2)

    layers = [layer1, layer2]

    net = Net(layers, use_last_layer=False)

    max_epoch = 20
    batch_size = 100
    data_size = len(contexts)
    max_iters = data_size // batch_size
    print('data size: {}, max iters: {}'.format(data_size, max_iters))

    start_time = time.time()
    loss_list = []
    for epoch in range(max_epoch):
        idx = np.random.permutation(np.arange(data_size))
        contexts = contexts[idx]
        target = target[idx]

        print("epoch: {} starts".format(epoch))
        for i in range(max_iters):

            contexts_batch = contexts[i * batch_size:(i + 1) * batch_size]
            target_batch = target[i * batch_size:(i + 1) * batch_size]

            kwargs_list = [{}, {layer2.positive_indexes_key: target_batch}]
            net.gradient_descent(contexts_batch, target_batch, kwargs_list=kwargs_list)

            loss = net.loss(contexts_batch, target_batch, kwargs_list=kwargs_list)
            if i % 100 == 0:
                print("epoch: {}, iter: {}, loss: {}".format(epoch, i, loss))
                loss_list.append(loss)

    print("execution time (s): %s" % (time.time() - start_time))
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    word_vec = layer1.params[0]
    target_words = ['you', 'year', 'car', 'ford']

    for target_word in target_words:
        word_similar_list = most_similar_words(target_word, wi_converter, word_vec)

        print('[target word]', target_word)
        for word, similarity in word_similar_list:
            print(word, ':', similarity)

        print('')

    plt.show()
