import time

import numpy as np
from matplotlib import pyplot as plt

from src.main.core.layer import Affine
from src.main.core import Net
from src.main.core import Adam
from src.main.language.layer import CBOWInputEmbedding, NegativeSampling
from src.main.language.util import process_text, ContextTargetConverter, WordIdConverter, convert_to_one_hot_encoding, \
    most_similar_words, UnigramSampler

if __name__ == '__main__':
    text = '''
    Importance of trade and commerce cannot be overstated; import and export activities drive the global economy. Countries import and export goods to meet their essential needs and promote economic growth. The import of raw materials is crucial for manufacturing, while the export of finished products can generate substantial revenue. Governments impose import tariffs and regulations to safeguard domestic industries, but these policies can impact international relations. In today's interconnected world, technology and information sharing are of utmost importance. The exchange of knowledge is as vital as physical imports and exports. As technology continues to advance, the importance of cybersecurity cannot be ignored, as it's necessary to protect against cyber threats. In a globalized world, understanding the intricate web of imports, exports, and information flows is key to thriving in the modern economy.
    '''

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]

    context_target_converter = ContextTargetConverter(word_id_list, window_size=1)
    contexts_original = context_target_converter.contexts()
    target_original = context_target_converter.targets()

    target = convert_to_one_hot_encoding(target_original, wi_converter.max_id())

    # build network
    input_layer_size = wi_converter.max_id() + 1
    hidden_layer_size = 50

    layer1 = CBOWInputEmbedding.from_size(input_layer_size, hidden_layer_size, Adam())
    layer2 = Affine.from_sizes(hidden_layer_size, input_layer_size, Adam(), useBias=False)

    layers = [layer1, layer2]

    net1 = Net(layers)

    init_weight1 = 1 / np.sqrt(input_layer_size)
    # init_weight1 = 0.01
    layer2_1 = CBOWInputEmbedding.from_size(input_layer_size, hidden_layer_size, Adam(), init_weight=init_weight1)
    sampler = UnigramSampler(word_id_list, remember_sampling=True)
    negative_size = 10
    init_weight2 = 1 / np.sqrt(hidden_layer_size)
    # init_weight2 = 0.01
    layer2_2 = NegativeSampling.from_size(hidden_layer_size, input_layer_size, negative_size, sampler, Adam(),
                                          init_weight=init_weight2)

    layers2 = [layer2_1, layer2_2]

    net2 = Net(layers2, use_last_layer=False)

    iter_num = 1000
    batch_size = 100
    loss_list1 = []
    loss_list2 = []

    start_time = time.time()
    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(contexts_original.shape[0], batch_size)
        x_batch_original = contexts_original[batch_mask]
        y_batch = target[batch_mask]
        y_batch_original = target_original[batch_mask]

        net1.gradient_descent(x_batch_original, y_batch)
        loss1 = net1.loss(x_batch_original, y_batch)
        loss_list1.append(loss1)

        kwargs_list = [{}, {layer2_2.positive_indexes_key: y_batch_original}]
        net2.gradient_descent(x_batch_original, y_batch_original, kwargs_list=kwargs_list)
        loss2 = net2.loss(x_batch_original, y_batch_original, kwargs_list=kwargs_list)
        loss_list2.append(loss2)

    print("execution time (s): %s" % (time.time() - start_time))
    # print(loss_list)
    x = list(range(1, len(loss_list1) + 1))
    y = loss_list1

    plt.plot(x, loss_list1, label='normal')
    plt.plot(x, loss_list2, label='negative sampling')
    plt.ylabel('loss')
    plt.legend()

    word_vec1 = layer1.params[0]
    word_vec2 = layer2_1.params[0]
    target_words = ['import', 'export']

    for target_word in target_words:
        word_similar_list1 = most_similar_words(target_word, wi_converter, word_vec1)
        word_similar_list2 = most_similar_words(target_word, wi_converter, word_vec2)

        print('[target word]', target_word)
        for i in range(len(word_similar_list1)):
            word1, similar1 = word_similar_list1[i]
            word2, similar2 = word_similar_list2[i]
            print('{} ({}) vs {} ({})'.format(word1, similar1, word2, similar2))

    plt.show()
