import random
from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import shuffle


class Sampler(Protocol):
    training: bool
    train_test_split_size: float

    def __call__(
        self, images: np.ndarray, labels: np.ndarray, classes: np.ndarray, imagings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def sample(self, images, labels, classes, imagings, train_index, test_index):
        if self.training:
            images, labels, classes, imagings = (
                images[train_index],
                labels[train_index],
                classes[train_index],
                imagings[train_index],
            )
        else:
            images, labels, classes, imagings = (
                images[test_index],
                labels[test_index],
                classes[test_index],
                imagings[test_index],
            )
        images, labels, classes, imagings = shuffle(images, labels, classes, imagings, random_state=0)
        return images, labels, classes, imagings

    @staticmethod
    def remove_list_items(in_list, to_remove):
        if not isinstance(to_remove, list):
            to_remove = [to_remove]
        return [item for item in in_list if item not in to_remove]

    @staticmethod
    def stratify_array(in_array, set_indices):
        random.seed(0)
        # display unique classes and calculate number of samples per each classes
        unique_classes, samples_per_class = np.unique(in_array[set_indices], return_counts=True)
        # min samples
        samples_min = samples_per_class.min()

        class_indices = []
        for cls_ in unique_classes:
            indices = np.where(in_array[set_indices] == cls_)[0]
            # under-sample without replacement
            indices = random.sample(indices.tolist(), samples_min)
            class_indices.append(indices)
        return np.sort(set_indices[np.concatenate(class_indices)])


####################################################################################################
# Examples of fundamental samplers (not used, just for showcase)


class RandomSampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes, imagings):
        idx_full = np.arange(len(images))

        np.random.seed(0)
        np.random.shuffle(idx_full)

        split = self.train_test_split_size
        len_test = int(len(images) * split)

        test_index = idx_full[0:len_test]
        train_index = np.delete(idx_full, np.arange(0, len_test))
        return self.sample(images, labels, classes, imagings, train_index, test_index)


class StratifySampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes, imagings):
        idx_full = np.arange(len(images))

        # note: train_test_split returns list containing train-test split of inputs.
        train_index, test_index = train_test_split(
            idx_full,
            test_size=self.train_test_split_size,
            stratify=classes,
            random_state=0,
            shuffle=True,
        )
        return self.sample(images, labels, classes, imagings, train_index, test_index)


class GroupSampler(Sampler):
    def __init__(self, training, train_test_split_size):
        self.training = training
        self.train_test_split_size = train_test_split_size

    def __call__(self, images, labels, classes, imagings):
        idx_full = np.arange(len(images))

        # note: GroupShuffleSplit generate indices to split data into training and test set.
        splitter = GroupShuffleSplit(test_size=self.train_test_split_size, n_splits=2, random_state=0)
        split = splitter.split(idx_full, groups=labels)
        train_index, test_index = next(split)
        return self.sample(images, labels, classes, imagings, train_index, test_index)


####################################################################################################


class BaseDumbSampler(Sampler):
    def __init__(self, training, train_test_split_size, variety_acronym, labels_to_remove):
        self.training = training
        self.train_test_split_size = train_test_split_size
        self.variety_acronym = variety_acronym
        self.labels_to_remove = labels_to_remove

    def __call__(self, images, labels, classes, imagings):
        labels_unique = np.unique(labels)

        # generate list by labels of treatment
        K_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [self.variety_acronym, "K"]
        ]
        S_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [self.variety_acronym, "S"]
        ]

        # remove to balance datasets
        K_list_unique = self.remove_list_items(K_list_unique, self.labels_to_remove["K"])
        S_list_unique = self.remove_list_items(S_list_unique, self.labels_to_remove["S"])
        
        # indices where labels correspond to each K or S treatment
        labels_K_indices = np.array([idx for idx, label in enumerate(labels) if label in K_list_unique])
        labels_S_indices = np.array([idx for idx, label in enumerate(labels) if label in S_list_unique])
        indices = np.concatenate((labels_K_indices, labels_S_indices))

        train_index, test_index = train_test_split(
            indices,
            test_size=self.train_test_split_size,
            random_state=0,
            shuffle=True,
        )

        # mess up classes
        classes_unique = np.unique(classes[indices])
        np.random.seed(0)
        classes = classes_unique[np.random.randint(0, len(classes_unique), (len(classes)))]

        # stratify by imagings train and test indices
        train_index = self.stratify_array(imagings, train_index)
        test_index = self.stratify_array(imagings, test_index)

        # stratify by classes train and test indices
        train_index = self.stratify_array(classes, train_index)
        test_index = self.stratify_array(classes, test_index)
        
        return self.sample(images, labels, classes, imagings, train_index, test_index)


class KrkaDumbSampler(BaseDumbSampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove)
        

class SavinjaDumbSampler(BaseDumbSampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove)


####################################################################################################


class BaseStratifySampler(Sampler):
    def __init__(self, training, train_test_split_size, variety_acronym, labels_to_remove):
        self.training = training
        self.train_test_split_size = train_test_split_size
        self.variety_acronym = variety_acronym
        self.labels_to_remove = labels_to_remove

    def __call__(self, images, labels, classes, imagings):
        labels_unique = np.unique(labels)

        # generate list by labels of treatment
        K_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [self.variety_acronym, "K"]
        ]
        S_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [self.variety_acronym, "S"]
        ]

        # remove to balance datasets
        K_list_unique = self.remove_list_items(K_list_unique, self.labels_to_remove["K"])
        S_list_unique = self.remove_list_items(S_list_unique, self.labels_to_remove["S"])

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

        # stratify by imagings train and test indices
        train_index = self.stratify_array(imagings, train_index)
        test_index = self.stratify_array(imagings, test_index)

        # stratify by classes train and test indices
        train_index = self.stratify_array(classes, train_index)
        test_index = self.stratify_array(classes, test_index)

        ## check the distribution, e.g. for imagings
        # unique, counts = np.unique(imagings[test_index], return_counts=True)
        # print(dict(zip(unique, counts)))

        return self.sample(images, labels, classes, imagings, train_index, test_index)


class KrkaStratifySampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove)


class SavinjaStratifySampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove)
