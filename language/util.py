import copy

import numpy as np


def process_text(text: str):
    # TODO: need to process other special characters
    return text.lower().strip().replace('.', ' .').replace(',', ' ,').split(' ')


def convert_to_one_hot_encoding(x: np.ndarray, max_value):
    original_shape = x.shape
    x = x.flatten()
    result = np.zeros((x.size, max_value + 1))
    result[np.arange(x.size), x] = 1

    return result.reshape(original_shape + (max_value + 1,))


class ConverterException(Exception):
    pass


class WordIdConverter:
    def __init__(self, words: list[str]):
        if len(words) == 0:
            raise ConverterException("The is no word in argument")

        self._original_words = words

        word_set = set()

        for word in self._original_words:
            word = word.strip()
            if len(word) > 0 and word not in word_set:
                word_set.add(word)

        sorted_word_list = sorted(list(word_set))

        self._word_length = len(sorted_word_list)

        self._word_to_id = {}
        self._id_to_word = {}
        self._max_id = 0
        for i in range(self._word_length):
            word = sorted_word_list[i]
            self._word_to_id[word] = i
            self._id_to_word[i] = word
            self._max_id = i

    def word_to_id(self, word: str) -> int:
        if word not in self._word_to_id:
            raise ConverterException("{} is not in converter data".format(word))
        return self._word_to_id[word]

    def id_to_word(self, id_: int) -> str:
        if id_ < 0:
            raise ConverterException("{0} should be positive integer".format(id_))

        if id_ >= self._word_length:
            raise ConverterException("{0} is larger than maximum id: {1}".format(id_, self._word_length - 1))

        return self._id_to_word[id_]

    def max_id(self) -> int:
        return self._max_id


class ContextTargetConverter:
    def __init__(self, word_ids: list[int], window_size=1):
        length = len(word_ids)
        minimum_length = window_size * 2 + 1
        if length < window_size * 2 + 1:
            raise ConverterException(
                'The minimum length of words should be at least {0} with window size {1}'.format(minimum_length,
                                                                                                 window_size))

        contexts = []
        targets = []
        for i in range(window_size, length - window_size):
            targets.append(word_ids[i])

            l_context = []
            r_context = []
            for j in range(window_size, 0, -1):
                l_context.append(word_ids[i - j])
                r_context.append(word_ids[i + j])

            r_context.reverse()
            contexts.append(l_context + r_context)

        self._contexts = np.array(contexts)
        self._targets = np.array(targets)

    def contexts(self) -> np.ndarray:
        return self._contexts

    def targets(self) -> np.ndarray:
        return self._targets


class UnigramExporter:
    def __init__(self, word_id_list: list[int], damp_coefficient=0.75):
        self._word_id_list = word_id_list

        length = len(word_id_list)
        max_id = max(word_id_list)
        word_id_length = max_id + 1
        count_by_id = np.zeros((word_id_length,))
        for word_id in word_id_list:
            count_by_id[word_id] += 1.0

        prob_by_id = count_by_id / length
        prob_by_id = np.power(prob_by_id, damp_coefficient)
        prob_by_id /= np.sum(prob_by_id)
        self._prob_by_id = prob_by_id
        self._id_array = np.arange(word_id_length)

    def export(self, size: int, exception_ids=None) -> np.ndarray:
        new_prob = self._prob_by_id

        if exception_ids:
            new_prob = copy.deepcopy(new_prob)
            new_prob[exception_ids] = 0.0
            new_prob /= np.sum(new_prob)

        result = np.random.choice(self._id_array, size=size, p=new_prob, replace=False)

        return result
