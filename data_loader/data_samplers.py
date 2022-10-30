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
        len_test = int(len(images) * split)

        test_index = idx_full[0:len_test]
        train_index = np.delete(idx_full, np.arange(0, len_test))

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
        # TODO: check if correct!
        train_index, test_index = next(split)

        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes


class ManualGroupSamplerKrka(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        labels_unique = np.unique(labels)

        # generate list by labels of treatment
        K_list = [label for label in labels_unique if label.split("-")[1] == "K"]
        S_list = [label for label in labels_unique if label.split("-")[1] == "S"]

        # remove because it has small number of samples
        K_list.remove("KK-K-09")

        # indices where labels correspond to each treatment
        labels_K_indices = np.array([idx for idx, label in enumerate(labels) if label in K_list])
        labels_S_indices = np.array([idx for idx, label in enumerate(labels) if label in S_list])

        # get labels for each treatment
        labels_K = labels[labels_K_indices]
        labels_S = labels[labels_S_indices]

        # split treatment separately
        splitter = GroupShuffleSplit(test_size=self.train_test_split_size, n_splits=2, random_state=0)
        # K
        split = splitter.split(labels_K_indices, groups=labels_K)
        train_idx, test_idx = next(split)
        train_idx_K, test_idx_K = labels_K_indices[train_idx], labels_K_indices[test_idx]
        # S
        split = splitter.split(labels_S_indices, groups=labels_S)
        train_idx, test_idx = next(split)
        train_idx_S, test_idx_S = labels_S_indices[train_idx], labels_S_indices[test_idx]

        # concatenate both
        train_index = np.concatenate((train_idx_K, train_idx_S))
        test_index = np.concatenate((test_idx_K, test_idx_S))

        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes
