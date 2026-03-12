import math
import random

from ragnarok.utils.data.dataset import Dataset


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool = False,
    ):
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        # math.ceil is used to ensure that we get the last batch even if it's smaller than batch_size
        self._max_iter = math.ceil(len(self._dataset) / self._batch_size)
        self._iter = None

        self._reset()

    def _reset(self):
        self._iter = 0

        self._indices = list(range(len(self._dataset)))
        if self._shuffle:
            random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter >= self._max_iter:
            self._reset()
            raise StopIteration

        start = self._iter * self._batch_size
        end = (self._iter + 1) * self._batch_size
        batch_indices = self._indices[start:end]

        batch_x = [self._dataset[i][0] for i in batch_indices]
        batch_t = [self._dataset[i][1] for i in batch_indices]

        self._iter += 1

        return batch_x, batch_t
