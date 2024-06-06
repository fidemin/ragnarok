from datetime import datetime

from src.main.data.sum_data_creator import create_sum_examples, convert_sum_examples_to_str


def test_create_sum_examples():
    seed = datetime.now().timestamp()

    number_of_examples = 1000
    min_int = 0
    max_int = 999
    actual = create_sum_examples(number_of_examples, min_int, max_int, seed)
    pair_checker = set()
    for first_term, second_term, sum_result in actual:
        assert (first_term + second_term) == sum_result
        assert min_int <= first_term <= max_int
        assert min_int <= second_term <= max_int
        pair_checker.add((first_term, second_term))

    assert len(pair_checker) == number_of_examples


def test_convert_sum_examples_to_str():
    input_ = [
        (10, 20, 30),
        (30, 234, 264),
        (40, 60, 100),
        (10, 900, 1000),
        (1, 2, 3)
    ]

    expected = [
        '10+20 =30  ',
        '30+234=264 ',
        '40+60 =100 ',
        '10+900=1000',
        '1+2   =3   '
    ]

    actual = convert_sum_examples_to_str(input_)

    assert len(actual) == len(expected)

    for i in range(len(actual)):
        assert actual[i] == expected[i]
