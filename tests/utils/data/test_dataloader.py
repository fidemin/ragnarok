from ragnarok.utils.data.dataloader import DataLoader
from ragnarok.utils.data.dataset import Dataset


class SimpleDataset(Dataset):
    def __init__(self, size: int):
        self._data = [(i, i * 2) for i in range(size)]

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


class TestDataLoader:
    def test_batch_size(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3)

        batch_x, batch_t = next(loader)

        assert len(batch_x) == 3
        assert len(batch_t) == 3

    def test_iteration_count(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3)

        batches = list(loader)

        assert len(batches) == 4

    def test_batch_values(self):
        dataset = SimpleDataset(5)
        loader = DataLoader(dataset, batch_size=5, shuffle=False)

        batch_x, batch_t = next(loader)

        assert batch_x == [0, 1, 2, 3, 4]
        assert batch_t == [0, 2, 4, 6, 8]

    def test_shuffle_changes_order(self):
        dataset = SimpleDataset(100)
        loader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
        loader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)

        no_shuffle_first, _ = next(loader_no_shuffle)
        shuffle_first, _ = next(loader_shuffle)

        assert no_shuffle_first != shuffle_first

    def test_reset_after_exhaustion(self):
        dataset = SimpleDataset(6)
        loader = DataLoader(dataset, batch_size=2)

        list(loader)

        # loader is already reset; next call starts a fresh epoch
        batch_x, _ = next(loader)
        assert len(batch_x) == 2

    def test_reusable_across_epochs(self):
        dataset = SimpleDataset(10)
        loader = DataLoader(dataset, batch_size=3)

        epoch1 = list(loader)
        epoch2 = list(loader)

        assert len(epoch1) == len(epoch2) == 4

    def test_iter_returns_self(self):
        dataset = SimpleDataset(4)
        loader = DataLoader(dataset, batch_size=2)

        assert iter(loader) is loader
