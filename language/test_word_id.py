import pytest

from language.word_id import WordIdConverter, ConverterException


def test_init():
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


def test_word_to_id():
    words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

    converter = WordIdConverter(words)

    assert converter.word_to_id('bye') == 0
    assert converter.word_to_id('good') == 1
    assert converter.word_to_id('i') == 2
    assert converter.word_to_id('say') == 3
    assert converter.word_to_id('you') == 4


def test_word_to_id_exception():
    words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

    converter = WordIdConverter(words)

    with pytest.raises(ConverterException):
        converter.word_to_id('abcd')


def test_id_to_word():
    words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

    converter = WordIdConverter(words)

    assert converter.id_to_word(0) == 'bye'
    assert converter.id_to_word(1) == 'good'
    assert converter.id_to_word(2) == 'i'
    assert converter.id_to_word(3) == 'say'
    assert converter.id_to_word(4) == 'you'


def test_id_to_word_exception():
    words = ['say', 'good', 'you', 'say', 'i', 'bye', 'bye', '']

    converter = WordIdConverter(words)

    with pytest.raises(ConverterException):
        converter.id_to_word(5)
