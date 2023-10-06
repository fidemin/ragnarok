class ConverterException(Exception):
    pass


class WordIdConverter:
    def __init__(self, words: list[str]):
        if len(words) == 0:
            raise ConverterException("The is no word in argument")

        self._words = words

        word_set = set()

        for word in self._words:
            word = word.strip()
            if len(word) > 0 and word not in word_set:
                word_set.add(word)

        sorted_word_list = sorted(list(word_set))

        self._word_length = len(sorted_word_list)

        self._word_to_id = {}
        self._id_to_word = {}
        for i in range(self._word_length):
            word = sorted_word_list[i]
            self._word_to_id[word] = i
            self._id_to_word[i] = word

    @classmethod
    def from_text(cls, text: str):
        # TODO: need to process other special characters
        words = text.lower().strip().replace('.', ' .').split(' ')
        return cls(words)

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
