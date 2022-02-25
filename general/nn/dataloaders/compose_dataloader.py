import random
from functools import reduce


class ComposeIterator:

    def __init__(self, iterators, shuffle=False):

        self.iterators = iterators
        self.shuffle = shuffle
        self._length = sum([len(it) for it in iterators])


    def __iter__(self):
        self.current_iterators = [
            iter(it) for it in self.iterators
        ]
        return self

    def __next__(self):

        while len(self.current_iterators) != 0:
            if self.shuffle:
                idx = random.choice(range(len(self.current_iterators)))
            else:
                idx = 0
            iterator = self.current_iterators[idx]
            try:
                return next(iterator)
            except StopIteration:
                del self.current_iterators[idx]
        raise StopIteration

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return self.iterators[idx]


class ComposeDataloaders(ComposeIterator):

    def __init__(self, iterators, shuffle=False):
        super().__init__(iterators, shuffle)

        datasets = [it.dataset for it in iterators]
        # self.dataset = sum(datasets[1:], datasets[0])
        self.dataset = reduce(lambda a, b: a+b, datasets)

    @property
    def batch_size(self):
        return self.iterators[0].batch_size
