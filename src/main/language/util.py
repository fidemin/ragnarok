import copy

import numpy as np


def process_text(text: str):
    # TODO: need to process other special characters
    return text.lower().strip().replace('.', ' .').replace(',', ' ,').replace('?', ' ?').replace('<eos>',
                                                                                                 ' <eos> ').split(' ')


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

    def ids_to_words(self, ids: list[int]) -> list[str]:
        return [self.id_to_word(id_) for id_ in ids]

    def max_id(self) -> int:
        return self._max_id

    def vocabulary_size(self) -> int:
        return self._max_id + 1


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


class UnigramSampler:
    def __init__(self, word_id_list: list[int], damp_coefficient=0.75, remember_sampling=False):
        self._word_id_list = word_id_list
        self._remember_sampling = remember_sampling
        self._result_cache = {}

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

    def sample(self, number_of_samples: int, sample_size: int, exception_ids=None) -> np.ndarray:
        if self._remember_sampling:
            key = self._cache_key(number_of_samples, sample_size, exception_ids)
            if self._result_cache.get(key) is not None:
                return self._result_cache[key]

        prob = self._prob_by_id

        result = np.zeros((number_of_samples, sample_size), dtype=int)

        for i in range(number_of_samples):
            if exception_ids is not None:
                prob = copy.deepcopy(self._prob_by_id)
                prob[exception_ids[i]] = 0.0
                prob /= np.sum(prob)

            result[i, :] = np.random.choice(self._id_array, size=sample_size, p=prob, replace=False)

        if self._remember_sampling:
            key = self._cache_key(number_of_samples, sample_size, exception_ids)
            self._result_cache[key] = result
        return result

    def _cache_key(self, number_of_samples: int, sample_size: int, exception_ids: np.ndarray) -> tuple:
        exception_ids_key = None
        if exception_ids is not None:
            exception_ids_key = exception_ids.tobytes()

        return number_of_samples, sample_size, exception_ids_key


def cosine_similarity(x: np.ndarray, y: np.ndarray):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def most_similar_words(word, word_id_converter: WordIdConverter, word_vec, top=5):
    word_id = word_id_converter.word_to_id(word)

    word_vec_row = word_vec[word_id]
    word_size = word_id_converter.max_id() + 1

    similarity = np.zeros(word_size)
    for i in range(word_size):
        similarity[i] = cosine_similarity(word_vec_row, word_vec[i])

    word_similarity_list = []
    count = 0
    for i in (-1 * similarity).argsort():
        if count >= top:
            break

        if i == word_id:
            continue

        word = word_id_converter.id_to_word(i)
        word_similarity_list.append((word, similarity[i]))

        count += 1

    return word_similarity_list
