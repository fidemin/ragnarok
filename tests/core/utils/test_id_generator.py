from ragnarok.core.utils.id_generator import IncrementalIdGenerator


class TestIncrementalIdGenerator:
    def test_incremental_id_generator(self):

        id_generator = IncrementalIdGenerator()
        assert id_generator.next() == 1
        assert id_generator.next() == 2
        assert id_generator.next() == 3
