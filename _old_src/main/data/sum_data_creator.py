import random as rand


def create_sum_examples(number_of_examples, min_int, max_int, seed=0):
    created_terms = set()

    rand.seed(seed)

    created = 0
    while number_of_examples > created:
        first_term = rand.randint(min_int, max_int)
        last_term = rand.randint(min_int, max_int)
        terms = (first_term, last_term)

        while terms in created_terms:
            first_term = rand.randint(min_int, max_int)
            last_term = rand.randint(min_int, max_int)
            terms = (first_term, last_term)

        created_terms.add(terms)
        created += 1

    result = [terms + (terms[0] + terms[1],) for terms in created_terms]
    return result


def convert_sum_examples_to_str(sum_examples: list[tuple]):
    max_left_size = 0
    max_right_size = 0

    sum_strs = []
    for first_term, second_term, sum_result in sum_examples:
        left_str = str(first_term) + '+' + str(second_term)
        right_str = '=' + str(sum_result)
        sum_strs.append((left_str, right_str))
        max_left_size = max(len(left_str), max_left_size)
        max_right_size = max(len(right_str), max_right_size)

    result = []
    for left_str, right_str in sum_strs:
        left_str = left_str.ljust(max_left_size)
        right_str = right_str.ljust(max_right_size)
        result.append(left_str + right_str)

    return result
