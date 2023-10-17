import copy

import numpy as np
import pytest

from language.util import WordIdConverter, ConverterException, ContextTargetConverter, convert_to_one_hot_encoding, \
    UnigramSampler, most_similar_words


@pytest.mark.parametrize(
    "test_input,max_id,expected",
    [
        (np.array([1, 2, 4, 3]), 5,
         np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0]])),

        (np.array([[0, 3], [2, 3], [1, 1]]), 3,
         np.array([[[1, 0, 0, 0], [0, 0, 0, 1]], [[0, 0, 1, 0], [0, 0, 0, 1]], [[0, 1, 0, 0], [0, 1, 0, 0]]])),
    ])
def test_convert_to_one_hot_encoding(test_input, max_id, expected):
    actual = convert_to_one_hot_encoding(test_input, max_id)
    np.array_equal(actual, expected)


class TestWordIdConverter:
    def test_init(self):
        words = ['say', 'say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

        converter = WordIdConverter(words)

        expected_length = 5
        expected_word_to_id = {
            'bye': 0,
            'good': 1,
            'i': 2,
            'say': 3,
            'you': 4
        }

        expected_id_to_word = {
            0: 'bye',
            1: 'good',
            2: 'i',
            3: 'say',
            4: 'you'
        }

        assert converter._word_length == 5
        assert len(converter._word_to_id) == len(expected_word_to_id)
        assert len(converter._id_to_word) == len(expected_id_to_word)

        for expected_k, expected_v in expected_word_to_id.items():
            assert converter._word_to_id[expected_k] == expected_v

        for expected_k, expected_v in expected_id_to_word.items():
            assert converter._id_to_word[expected_k] == expected_v

    def test_word_to_id(self):
        words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

        converter = WordIdConverter(words)

        assert converter.word_to_id('bye') == 0
        assert converter.word_to_id('good') == 1
        assert converter.word_to_id('i') == 2
        assert converter.word_to_id('say') == 3
        assert converter.word_to_id('you') == 4

    def test_word_to_id_exception(self):
        words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

        converter = WordIdConverter(words)

        with pytest.raises(ConverterException):
            converter.word_to_id('abcd')

    def test_id_to_word(self):
        words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

        converter = WordIdConverter(words)

        assert converter.id_to_word(0) == 'bye'
        assert converter.id_to_word(1) == 'good'
        assert converter.id_to_word(2) == 'i'
        assert converter.id_to_word(3) == 'say'
        assert converter.id_to_word(4) == 'you'

    def test_id_to_word_exception(self):
        words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

        converter = WordIdConverter(words)

        with pytest.raises(ConverterException):
            converter.id_to_word(5)


class TestContextTargetConverter:
    @pytest.mark.parametrize(
        "test_input,window_size,expected_contexts,expected_targets",
        [
            ([0, 1, 2, 3, 4, 1, 5, 1], 1, [[0, 2], [1, 3], [2, 4], [3, 1], [4, 5], [1, 1]], [1, 2, 3, 4, 1, 5]),
            ([0, 1, 2, 3, 4, 1, 5, 1], 2, [[0, 1, 3, 4], [1, 2, 4, 1], [2, 3, 1, 5], [3, 4, 5, 1]], [2, 3, 4, 1]),
        ])
    def test_init_contexts_targets(self, test_input, window_size, expected_contexts, expected_targets):
        converter = ContextTargetConverter(test_input, window_size=window_size)
        assert np.array_equal(converter.contexts(), expected_contexts)
        assert np.array_equal(converter.targets(), expected_targets)


class TestUnigramSampler:
    def test_init(self):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 2, 3]
        dc = 0.75
        sum_ = 0.2 ** dc + 0.3 ** dc + 0.2 ** dc + 0.2 ** dc + 0.0 ** dc + 0.1 ** dc
        expected_prob_by_id = np.array(
            [(0.2 ** dc) / sum_, (0.3 ** dc) / sum_, (0.2 ** dc) / sum_, (0.2 ** dc) / sum_, (0.0 ** dc) / sum_,
             (0.1 ** dc) / sum_])

        exporter = UnigramSampler(word_id_list, damp_coefficient=dc)
        actual = exporter._prob_by_id
        assert np.allclose(actual, expected_prob_by_id)

    def test_sample(self):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 2, 3]
        max_id = 5
        dc = 0.75
        exporter = UnigramSampler(word_id_list, damp_coefficient=dc)
        original_prob_by_id = copy.deepcopy(exporter._prob_by_id)
        number_of_samples = 4
        size = 2

        for i in range(1000):
            actual = exporter.sample(number_of_samples, size)
            assert actual.shape == (number_of_samples, size)
            assert np.all(np.where(np.logical_and(actual >= 0, actual <= max_id), True, False))

        for i in range(1000):
            exception_ids = np.array([1, 3, 2, 4])
            actual = exporter.sample(number_of_samples, size, exception_ids=exception_ids)
            assert actual.shape == (number_of_samples, size)

            for i, exception_id in enumerate(exception_ids):
                condition = np.logical_and(np.logical_and(actual[i] >= 0, actual[i] <= max_id),
                                           np.logical_and(actual[i] != exception_id, True))
                assert np.all(np.where(condition, True, False))
                assert np.allclose(exporter._prob_by_id, original_prob_by_id)

    @pytest.mark.parametrize(
        "number_of_samples,sample_size,exception_ids",
        [
            (4, 2, None),
            (4, 2, np.array([1, 3, 2, 4])),
        ])
    def test_sample_remember_sampling(self, number_of_samples, sample_size, exception_ids):
        word_id_list = [0, 1, 2, 3, 0, 1, 5, 1, 2, 3]
        dc = 0.75
        exporter = UnigramSampler(word_id_list, damp_coefficient=dc, remember_sampling=True)

        exception_ids = np.array([1, 3, 2, 4])
        last_actual = exporter.sample(number_of_samples, sample_size, exception_ids=exception_ids)
        actual = exporter.sample(number_of_samples, sample_size, exception_ids=exception_ids)

        diff = exporter.sample(4, 2, np.array([2, 3, 4, 2]))
        assert np.allclose(actual, last_actual)
        assert not np.allclose(actual, diff)


def test_most_similar():
    words = ['say', 'say', 'good', 'you', 'say', 'i', 'bye', 'bye']

    converter = WordIdConverter(words)

    word_vec = np.array([
        # target word is 'bye', word_id=0
        [0.1, 0.2, 0.1, 0.2, 0.0, 0.7],
        [0.2, 0.1, -0.1, 0.3, 0.0, 0.8],
        # exact same vec with 1st row
        [0.1, 0.2, 0.1, 0.2, 0.0, 0.7],
        [-0.1, -0.2, -1.3, 1.0, 0.0, 1.3],
        [1.1, 0.1, 2.2, 3.3, 1.1, 2.3]
    ])

    word_similar_list = most_similar_words(converter.id_to_word(0), converter, word_vec, top=2)
    assert word_similar_list[0][0] == converter.id_to_word(2)
