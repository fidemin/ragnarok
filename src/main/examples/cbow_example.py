import time

import numpy as np
from matplotlib import pyplot as plt

from src.main.core.layer import Affine
from src.main.core import Net
from src.main.core import Adam
from src.main.language.layer import CBOWInput
from src.main.language.util import process_text, ContextTargetConverter, WordIdConverter, convert_to_one_hot_encoding

if __name__ == '__main__':
    text = '''
    Importance of trade and commerce cannot be overstated; import and export activities drive the global economy. Countries import and export goods to meet their essential needs and promote economic growth. The import of raw materials is crucial for manufacturing, while the export of finished products can generate substantial revenue. Governments impose import tariffs and regulations to safeguard domestic industries, but these policies can impact international relations. In today's interconnected world, technology and information sharing are of utmost importance. The exchange of knowledge is as vital as physical imports and exports. As technology continues to advance, the importance of cybersecurity cannot be ignored, as it's necessary to protect against cyber threats. In a globalized world, understanding the intricate web of imports, exports, and information flows is key to thriving in the modern economy.
    '''

    words = process_text(text)
    wi_converter = WordIdConverter(words)
    word_id_list = [wi_converter.word_to_id(word) for word in words]

    context_target_converter = ContextTargetConverter(word_id_list)
    contexts = context_target_converter.contexts()
    target = context_target_converter.targets()

    contexts = convert_to_one_hot_encoding(contexts, wi_converter.max_id())
    target = convert_to_one_hot_encoding(target, wi_converter.max_id())

    # build network
    input_layer_size = wi_converter.max_id() + 1
    hidden_layer_size = 50

    layer1 = CBOWInput.from_size(input_layer_size, hidden_layer_size, Adam())
    layer2 = Affine.from_sizes(hidden_layer_size, input_layer_size, Adam(), useBias=False)

    layers = [layer1, layer2]

    net = Net(layers)

    iter_num = 2000
    batch_size = 100
    loss_list = []
    start_time = time.time()
    for i in range(iter_num):
        print("iter_num: {} starts".format(i))
        batch_mask = np.random.choice(contexts.shape[0], batch_size)
        x_batch = contexts[batch_mask]
        y_batch = target[batch_mask]

        net.gradient_descent(x_batch, y_batch)

        loss = net.loss(x_batch, y_batch)
        loss_list.append(loss)

    print("execution time (s): %s" % (time.time() - start_time))
    # print(loss_list)
    y = loss_list
    x = list(range(1, len(loss_list) + 1))

    plt.plot(x, y)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.show()
