import numpy as np
import pytest

from language.util import WordIdConverter, ConverterException, ContextTargetConverter, convert_to_one_hot_encoding


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
