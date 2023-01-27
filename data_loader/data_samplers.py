import random
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

    def sample(self, images, classes, train_index, test_index):
        if self.training:
            images, classes = images[train_index], classes[train_index]
        else:
            images, classes = images[test_index], classes[test_index]
        images, classes = shuffle(images, classes, random_state=0)
        return images, classes


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
        return self.sample(images, classes, train_index, test_index)


class StratifySampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        idx_full = np.arange(len(images))

        # note: train_test_split returns list containing train-test split of inputs.
        train_index, test_index = train_test_split(
            idx_full,
            test_size=self.train_test_split_size,
            stratify=classes,
            random_state=0,
            shuffle=True,
        )
        return self.sample(images, classes, train_index, test_index)


class GroupSampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        idx_full = np.arange(len(images))

        # note: GroupShuffleSplit generate indices to split data into training and test set.
        splitter = GroupShuffleSplit(test_size=self.train_test_split_size, n_splits=2, random_state=0)
        split = splitter.split(idx_full, groups=labels)
        train_index, test_index = next(split)
        return self.sample(images, classes, train_index, test_index)


class ManualGroupSamplerKrka(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes):
        labels_unique = np.unique(labels)

        # generate list by labels of treatment
        K_list_unique = [label for label in labels_unique if label.split("-")[:2] == ['KK', 'K']]
        S_list_unique = [label for label in labels_unique if label.split("-")[:2] == ['KK', 'S']]

        # remove because it has small number of samples, and to have same K and S
        K_list_unique.remove("KK-K-09")

        # indices where labels correspond to each K or S treatment
        labels_K_indices = np.array([idx for idx, label in enumerate(labels) if label in K_list_unique])
        labels_S_indices = np.array([idx for idx, label in enumerate(labels) if label in S_list_unique])

        # get labels for each treatment
        labels_K = labels[labels_K_indices]
        labels_S = labels[labels_S_indices]

        # split treatments separately
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

        # stratify by classes train and test indices
        train_index = self.stratify_classes(classes, train_index)
        test_index = self.stratify_classes(classes, test_index)

        return self.sample(images, classes, train_index, test_index)

    @staticmethod
    def stratify_classes(classes, set_indices):
        random.seed(0)
        # display unique classes and calculate number of samples per each classes
        unique_classes, samples_per_class = np.unique(classes[set_indices], return_counts=True)
        # min samples
        samples_min = samples_per_class.min()

        class_indices = []
        for cls_ in unique_classes:
            indices = np.where(classes[set_indices] == cls_)[0]
            # under-sample without replacement
            indices = random.sample(indices.tolist(), samples_min)
            class_indices.append(indices)
        return np.sort(set_indices[np.concatenate(class_indices)])

