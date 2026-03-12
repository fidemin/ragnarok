class Dataset:
    def __getitem__(self, index):
        raise NotImplementedError("Subclasses must implement __getitem__ method.")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__ method.")
