from datetime import datetime

import numpy as np

from data.sum_data_creator import create_sum_examples, convert_sum_examples_to_str
from language.util import WordIdConverter


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
    return training_X, training_y, test_X, test_y


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_data(50000, 0, 999, 0.02)
    print('train X: {}, train y: {}, test X: {}, text y: {}'.format(train_X.shape, train_y.shape, test_X.shape,
                                                                    test_y.shape))
