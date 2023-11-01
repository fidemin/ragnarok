from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from core.learning_rate import InverseSqrt
from core.net import NeuralNet
from core.optimizer import SGD
from data.sum_data_creator import create_sum_examples, convert_sum_examples_to_str
from language.layer import GroupedSoftmaxWithLoss
from language.seq2seq import Seq2Seq
from language.util import WordIdConverter, convert_to_one_hot_encoding


def load_data(number_of_examples, min_int, max_int, proportion_of_test=0.1):
    seed = datetime.now().timestamp()
    sum_examples_original = create_sum_examples(number_of_examples, min_int, max_int, seed=seed)
    sum_str_examples = convert_sum_examples_to_str(sum_examples_original)
    sum_char_examples = [list(sum_str) for sum_str in sum_str_examples]

    sum_char_all_list = []
    for char_list in sum_char_examples:
        sum_char_all_list.extend(char_list)

    wi_converter = WordIdConverter(sum_char_all_list)

    left_sum_char_examples = []
    right_sum_char_examples = []
    for sum_str in sum_str_examples:
        left, right = sum_str.split('=')
        right = '=' + right
        left_sum_char_examples.append(list(left))
        right_sum_char_examples.append(list(right))

    left_sum_id_examples = [[wi_converter.word_to_id(char_) for char_ in char_list] for char_list in
                            left_sum_char_examples]
    right_sum_id_examples = [[wi_converter.word_to_id(char_) for char_ in char_list] for char_list in
                             right_sum_char_examples]
    left_sum_id_examples = np.array(left_sum_id_examples)
    right_sum_id_examples = np.array(right_sum_id_examples)
    assert left_sum_id_examples.shape[0] == right_sum_id_examples.shape[0]

    length = left_sum_id_examples.shape[0]
    training_X, test_X = np.split(left_sum_id_examples, [int(length * (1 - proportion_of_test))])
    training_y, test_y = np.split(right_sum_id_examples, [int(length * (1 - proportion_of_test))])
    return training_X, training_y, test_X, test_y, wi_converter


def convert_decode_input_to_expected_output(decoder_input: np.ndarray, max_word_id: int, one_hot=True):
    expected_output = np.delete(train_y, 0, axis=1)
    expected_output = np.insert(expected_output, expected_output.shape[1], values=word_id_blank, axis=1)

    if one_hot:
        expected_output = convert_to_one_hot_encoding(expected_output, wi_converter.max_id())
    return expected_output


if __name__ == '__main__':
    train_X, train_y, test_X, test_y, wi_converter = load_data(50000, 0, 999, 0.02)
    print('train X: {}, train y: {}, test X: {}, text y: {}'.format(
        train_X.shape, train_y.shape, test_X.shape, test_y.shape))

    word_id_blank = wi_converter.word_to_id(' ')
    word_id_equal = wi_converter.word_to_id('=')

    train_t = convert_decode_input_to_expected_output(train_y, wi_converter.max_id(), one_hot=True)
    test_t_no_one_hot = convert_decode_input_to_expected_output(test_y, wi_converter.max_id(), one_hot=False)
    test_t = convert_to_one_hot_encoding(test_t_no_one_hot, wi_converter.max_id())
    t_size = test_t.shape[1]

    voca_size = wi_converter.vocabulary_size()
    wordvec_size = 100
    hidden_size = 100
    max_epoch = 20
    data_size = train_X.shape[0]
    batch_size = 100
    iter_size = data_size // batch_size

    seq2seq = Seq2Seq(voca_size, wordvec_size, hidden_size)
    loss_layer = GroupedSoftmaxWithLoss()

    net = NeuralNet([seq2seq], loss_layer=loss_layer, optimizer=SGD(InverseSqrt(5.0)))

    loss_list = []
    correct_ratios = []
    for epoch in range(1, max_epoch + 1):
        idx = np.random.permutation(np.arange(data_size))
        train_X = train_X[idx]
        train_y = train_y[idx]
        train_t = train_t[idx]

        for iter_ in range(iter_size):
            mini_train_X = train_X[iter_ * batch_size: (iter_ + 1) * batch_size]
            mini_train_y = train_y[iter_ * batch_size: (iter_ + 1) * batch_size]
            mini_train_t = train_t[iter_ * batch_size: (iter_ + 1) * batch_size]

            kwarg_list = [{Seq2Seq.decoder_xs_key: mini_train_y}]
            loss = net.forward(mini_train_X, mini_train_t, kwargs_list=kwarg_list)
            loss_list.append(loss)
            net.backward()
            net.optimize(grad_max_norm=0.5, epoch=epoch)

            if iter_ % 100 == 0:
                print('iter: {}, loss: {}'.format(iter_, loss))

            loss_list.append(loss)

        correct = 0
        print('epoch {}'.format(epoch))
        for i in range(test_X.shape[0]):
            result = seq2seq.generate(test_X[[i]], word_id_equal, t_size)
            expected = test_t_no_one_hot[i]
            expected_list = expected.tolist()

            # print(''.join(wi_converter.ids_to_words(result)))
            # print(''.join(wi_converter.ids_to_words(expected_list)))

            if i < 10:
                input_sum = ''.join(wi_converter.ids_to_words(test_X[i].tolist()))
                actual_output = ''.join(wi_converter.ids_to_words(result))
                expected_output = ''.join(wi_converter.ids_to_words(expected_list))
                print('-------------')
                print('input: {}'.format(input_sum))
                print('expected: {}'.format(expected_output))
                print('actual: {}'.format(actual_output))

            if np.allclose(np.array(result), expected):
                correct += 1

        correct_ratio = correct / test_X.shape[0]
        print('epoch {}: correct ratio: {}'.format(epoch, correct_ratio))
        correct_ratios.append(correct_ratio)

    plt.subplot(2, 1, 1)
    loss_x = list(range(1, len(loss_list) + 1))
    plt.plot(loss_x, loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(2, 1, 2)
    cr_x = list(range(1, len(correct_ratios) + 1))
    plt.plot(cr_x, correct_ratios)
    plt.xlabel('epoch')
    plt.ylabel('correct ratio')

    plt.show()
