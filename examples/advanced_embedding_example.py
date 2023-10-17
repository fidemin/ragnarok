import time

import numpy as np
from matplotlib import pyplot as plt

from core.net import Net
from core.updater import Adam
from language.layer import CBOWInputEmbedding, NegativeSampling
from language.util import process_text, ContextTargetConverter, WordIdConverter, UnigramSampler, most_similar_words

if __name__ == '__main__':
    # text = '''
    # Importance of trade and commerce cannot be overstated; import and export activities drive the global economy. Countries import and export goods to meet their essential needs and promote economic growth. The import of raw materials is crucial for manufacturing, while the export of finished products can generate substantial revenue. Governments impose import tariffs and regulations to safeguard domestic industries, but these policies can impact international relations. In today's interconnected world, technology and information sharing are of utmost importance. The exchange of knowledge is as vital as physical imports and exports. As technology continues to advance, the importance of cybersecurity cannot be ignored, as it's necessary to protect against cyber threats. In a globalized world, understanding the intricate web of imports, exports, and information flows is key to thriving in the modern economy.
    # '''
    text = 'You say no. I say no. You said yes. I said yes. Kim said yes. Min said no.'
    text = 'I say yes. I said yes. You said yes. You say yes.'

    # the file is from: https://github.com/wojzaremba/lstm/blob/master/data/ptb.train.txt
    # file_path = '../downloads/ptb.train.txt'
    # text = ''
    # with open(file_path) as f:
    #     text = f.read().replace('\n', '<eos>').strip()

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]

    # window_size = 5
    window_size = 1
    context_target_converter = ContextTargetConverter(word_id_list, window_size=window_size)
    contexts = context_target_converter.contexts()
    target = context_target_converter.targets()

    id_to_word_vec = np.vectorize(wi_converter.id_to_word)
    for i in range(target.shape[0]):
        print(id_to_word_vec(contexts[i]))
        print(id_to_word_vec(target[i]))

    # build network
    input_layer_size = wi_converter.max_id() + 1
    hidden_layer_size = 100

    init_weight1 = 1 / np.sqrt(input_layer_size)
    # init_weight1 = 0.01
    layer1 = CBOWInputEmbedding.from_size(input_layer_size, hidden_layer_size, Adam(), init_weight=init_weight1)
    sampler = UnigramSampler(word_id_list)
    negative_size = 2
    # init_weight2 = 1 / np.sqrt(hidden_layer_size)
    init_weight2 = 0.01
    layer2 = NegativeSampling.from_size(hidden_layer_size, input_layer_size, negative_size, sampler, Adam(),
                                        init_weight=init_weight2)

    layers = [layer1, layer2]

    net = Net(layers, use_last_layer=False)

    max_epoch = 100
    # batch_size = 100
    batch_size = 1
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
            if i % 100 == 0:
                print("epoch: {}, iter: {} starts".format(epoch, i))
            contexts_batch = contexts[i * batch_size:(i + 1) * batch_size]
            target_batch = target[i * batch_size:(i + 1) * batch_size]

            kwargs_list = [{}, {layer2.positive_indexes_key: target_batch}]
            net.gradient_descent(contexts_batch, target_batch, kwargs_list=kwargs_list)

            loss = net.loss(contexts_batch, target_batch, kwargs_list=kwargs_list)

        loss_list.append(loss)

    print("execution time (s): %s" % (time.time() - start_time))
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('epoch')
    plt.ylabel('loss')

    word_vec = layer1.params[0]
    # target_words = ['you', 'year', 'car', 'ford']
    target_words = ['said']

    for target_word in target_words:
        word_similar_list = most_similar_words(target_word, wi_converter, word_vec)

        print('[target word]', target_word)
        for word, similarity in word_similar_list:
            print(word, ':', similarity)

        print('')

    plt.show()
