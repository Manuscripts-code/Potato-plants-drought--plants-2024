from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import shuffle


class Sampler(Protocol):
    training: bool
    train_test_split_size: float

    def __call__(
        self, images: np.ndarray, labels: np.ndarray, classes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


class RandomSampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        idx_full = np.arange(len(images))

        np.random.seed(0)
        np.random.shuffle(idx_full)

        split = self.train_test_split_size
        len_valid = int(len(images) * split)

        test_index = idx_full[0:len_valid]
        train_index = np.delete(idx_full, np.arange(0, len_valid))

        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes


class StratifySampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        idx_full = np.arange(len(images))

        train_index, test_index = train_test_split(
            idx_full,
            test_size=self.train_test_split_size,
            stratify=classes,
            random_state=0,
            shuffle=True,
        )

        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes


class GroupSampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        idx_full = np.arange(len(images))
        np.random.seed(0)
        np.random.shuffle(idx_full)

        splitter = GroupShuffleSplit(test_size=self.train_test_split_size, n_splits=2, random_state=0)
        split = splitter.split(idx_full, groups=labels)
        train_index, test_index = next(split)

        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes
