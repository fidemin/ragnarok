class IncrementalIdGenerator:
    def __init__(self):
        self._id = 0

    def next(self):
        self._id += 1
        return self._id
