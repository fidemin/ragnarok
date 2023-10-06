from language.word_id import WordIdConverter


def test_init():
    words = ['say', 'good', 'you', 'say', 'i', 'bye']

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
