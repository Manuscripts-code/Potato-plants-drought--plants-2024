import itertools
import random
from collections import namedtuple
from typing import Protocol

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import shuffle


class Sampler(Protocol):
    training: bool
    train_test_split_size: float
    Distribution = namedtuple("Distribution", ["share_I", "share_C", "share_D"])

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

        # indc = train_index[np.where(imagings[test_index] == "imaging-3")[0]]
        # unique, counts = np.unique(classes[indc], return_counts=True)
        # print(dict(zip(unique, counts)))

    @staticmethod
    def _get_max_samples(imagings, classes, indices):
        imagings_unique = np.unique(imagings[indices])

        counts_all = []
        for imaging in imagings_unique:
            indices_imaging = indices[np.where(imagings[indices] == imaging)[0]]
            _, counts = np.unique(classes[indices_imaging], return_counts=True)
            counts_all.extend(counts)

        return min(counts_all)

    @staticmethod
    def _apply_distribution(distributions, imaging, class_, samples_num_max):
        if distributions is None:
            return samples_num_max

        distribution = distributions[imaging]
        if class_.split("_")[-1] == "control":
            share_class = distribution.share_C
        else:
            share_class = distribution.share_D

        return int(samples_num_max * share_class * distribution.share_I)

    @staticmethod
    def resample_indices(imagings, classes, indices, distributions=None):
        random.seed(0)
        imagings_unique = np.unique(imagings[indices])
        classes_unique = np.unique(classes[indices])

        samples_num_max = Sampler._get_max_samples(imagings, classes, indices)

        indices_resampled = []
        for imaging, class_ in itertools.product(imagings_unique, classes_unique):
            indices_sep = np.where((imagings[indices] == imaging) & (classes[indices] == class_))[0]
            samples_num = Sampler._apply_distribution(distributions, imaging, class_, samples_num_max)
            # under-sample without replacement
            indices_sep = random.sample(indices_sep.tolist(), samples_num)
            indices_resampled.append(indices_sep)

        return np.sort(indices[np.concatenate(indices_resampled).astype("int")])

    @staticmethod
    def separate_labels_by_treatment(labels, labels_to_remove, variety_acronym):
        labels_unique = np.unique(labels)

        # generate list by labels of treatment
        K_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [variety_acronym, "K"]
        ]
        S_list_unique = [
            label for label in labels_unique if label.split("-")[:2] == [variety_acronym, "S"]
        ]

        # remove to balance datasets
        K_list_unique = Sampler.remove_list_items(K_list_unique, labels_to_remove["K"])
        S_list_unique = Sampler.remove_list_items(S_list_unique, labels_to_remove["S"])

        # indices where labels correspond to each K or S treatment
        labels_K_indices = np.array([idx for idx, label in enumerate(labels) if label in K_list_unique])
        labels_S_indices = np.array([idx for idx, label in enumerate(labels) if label in S_list_unique])

        return labels_K_indices, labels_S_indices

    @staticmethod
    def create_plant_test_split(labels, labels_K_indices, labels_S_indices, train_test_split_size):
        # get labels for each treatment
        labels_K = labels[labels_K_indices]
        labels_S = labels[labels_S_indices]

        # split treatments separately
        splitter = GroupShuffleSplit(test_size=train_test_split_size, n_splits=2, random_state=0)
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

        return train_index, test_index


####################################################################################################
# Examples of fundamental samplers (not used, just for showcase)


# class RandomSampler(Sampler):
#     def __init__(self, training, train_test_split_size):
#         self.training = training
#         self.train_test_split_size = train_test_split_size

#     def __call__(self, images, labels, classes, imagings):
#         idx_full = np.arange(len(images))

#         np.random.seed(0)
#         np.random.shuffle(idx_full)

#         split = self.train_test_split_size
#         len_test = int(len(images) * split)

#         test_index = idx_full[0:len_test]
#         train_index = np.delete(idx_full, np.arange(0, len_test))
#         return self.sample(images, labels, classes, imagings, train_index, test_index)


# class StratifySampler(Sampler):
#     def __init__(self, training, train_test_split_size):
#         self.training = training
#         self.train_test_split_size = train_test_split_size

#     def __call__(self, images, labels, classes, imagings):
#         idx_full = np.arange(len(images))

#         # note: train_test_split returns list containing train-test split of inputs.
#         train_index, test_index = train_test_split(
#             idx_full,
#             test_size=self.train_test_split_size,
#             stratify=classes,
#             random_state=0,
#             shuffle=True,
#         )
#         return self.sample(images, labels, classes, imagings, train_index, test_index)


# class GroupSampler(Sampler):
#     def __init__(self, training, train_test_split_size):
#         self.training = training
#         self.train_test_split_size = train_test_split_size

#     def __call__(self, images, labels, classes, imagings):
#         idx_full = np.arange(len(images))

#         # note: GroupShuffleSplit generate indices to split data into training and test set.
#         splitter = GroupShuffleSplit(test_size=self.train_test_split_size, n_splits=2, random_state=0)
#         split = splitter.split(idx_full, groups=labels)
#         train_index, test_index = next(split)
#         return self.sample(images, labels, classes, imagings, train_index, test_index)


####################################################################################################
# Simple samplers do not take into consideration the labels of the images (non plant splits)


class BaseSimpleSampler(Sampler):
    def __init__(self, training, train_test_split_size, variety_acronym, labels_to_remove, dumb):
        self.training = training
        self.train_test_split_size = train_test_split_size
        self.variety_acronym = variety_acronym
        self.labels_to_remove = labels_to_remove
        self.dumb = dumb

    def __call__(self, images, labels, classes, imagings):
        labels_K_indices, labels_S_indices = self.separate_labels_by_treatment(
            labels, self.labels_to_remove, self.variety_acronym
        )
        indices = np.concatenate((labels_K_indices, labels_S_indices))

        # split does not take in consideration the underlying labels
        train_index, test_index = train_test_split(
            indices,
            test_size=self.train_test_split_size,
            random_state=0,
            shuffle=True,
        )
        if self.dumb:
            # mess up classes if dumb is enabled
            classes_unique = np.unique(classes[indices])
            np.random.seed(0)
            classes = classes_unique[np.random.randint(0, len(classes_unique), (len(classes)))]

        # stratify by imagings and classes for train and test indices
        train_index = self.resample_indices(imagings, classes, indices=train_index)
        test_index = self.resample_indices(imagings, classes, indices=test_index)

        return self.sample(images, labels, classes, imagings, train_index, test_index)


# Dumb sampler for testing purposes (classes are shuffled)
class KrkaDumbSampler(BaseSimpleSampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        dumb = True
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove, dumb)


# Simple samplers which do not take into consideration the labels of the images
# slices arbitrarily distributed into train and test sets
class KrkaRandomSampler(BaseSimpleSampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        dumb = False
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove, dumb)


class SavinjaRandomSampler(BaseSimpleSampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        dumb = False
        super().__init__(training, train_test_split_size, variety_acronym, labels_to_remove, dumb)


####################################################################################################
# Stratify samplers sample the data in correct way (plants train/test splits)
# Stratification is done by labels, classes and imagings.


class BaseStratifySampler(Sampler):
    def __init__(
        self, training, train_test_split_size, variety_acronym, labels_to_remove, distributions
    ):
        self.training = training
        self.train_test_split_size = train_test_split_size
        self.variety_acronym = variety_acronym
        self.labels_to_remove = labels_to_remove
        self.distributions = distributions

    def __call__(self, images, labels, classes, imagings):
        labels_K_indices, labels_S_indices = self.separate_labels_by_treatment(
            labels, self.labels_to_remove, self.variety_acronym
        )

        train_index, test_index = self.create_plant_test_split(
            labels, labels_K_indices, labels_S_indices, self.train_test_split_size
        )

        # stratify by imagings and classes for train and test indices
        train_index = self.resample_indices(
            imagings, classes, indices=train_index, distributions=self.distributions
        )
        test_index = self.resample_indices(
            imagings, classes, indices=test_index, distributions=self.distributions
        )

        ## check the distribution, e.g. for imagings
        # unique, counts = np.unique(imagings[test_index], return_counts=True)
        # print(dict(zip(unique, counts)))

        return self.sample(images, labels, classes, imagings, train_index, test_index)


class KrkaStratifySampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        distributions = None
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )


class SavinjaStratifySampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        distributions = None
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )

####################################################################################################
# Samplers with pre-defined biases


class KrkaBiasedImagingsSampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        distributions = {
            "imaging-1": self.Distribution(share_I=0.2, share_C=1, share_D=1),
            "imaging-2": self.Distribution(share_I=0.2, share_C=1, share_D=1),
            "imaging-3": self.Distribution(share_I=0.6, share_C=1, share_D=1),
            "imaging-4": self.Distribution(share_I=1, share_C=1, share_D=1),
            "imaging-5": self.Distribution(share_I=1, share_C=1, share_D=1),
        }
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )


class KrkaBiasedTreatmentSampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KK"
        labels_to_remove = {"K": "KK-K-09", "S": "KK-S-01"}
        distributions = {
            "imaging-1": self.Distribution(share_I=1, share_C=1, share_D=0.2),
            "imaging-2": self.Distribution(share_I=1, share_C=1, share_D=0.2),
            "imaging-3": self.Distribution(share_I=1, share_C=0.6, share_D=0.6),
            "imaging-4": self.Distribution(share_I=1, share_C=0.2, share_D=1),
            "imaging-5": self.Distribution(share_I=1, share_C=0.2, share_D=1),
        }
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )


class SavinjaBiasedImagingsSampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        distributions = {
            "imaging-1": self.Distribution(share_I=0.2, share_C=1, share_D=1),
            "imaging-2": self.Distribution(share_I=0.2, share_C=1, share_D=1),
            "imaging-3": self.Distribution(share_I=0.6, share_C=1, share_D=1),
            "imaging-4": self.Distribution(share_I=1, share_C=1, share_D=1),
            "imaging-5": self.Distribution(share_I=1, share_C=1, share_D=1),
        }
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )


class SavinjaBiasedTreatmentSampler(BaseStratifySampler):
    def __init__(self, training, train_test_split_size):
        variety_acronym = "KS"
        labels_to_remove = {"K": "KS-K-15", "S": ["KS-S-04", "KS-S-12"]}
        distributions = {
            "imaging-1": self.Distribution(share_I=1, share_C=1, share_D=0.2),
            "imaging-2": self.Distribution(share_I=1, share_C=1, share_D=0.2),
            "imaging-3": self.Distribution(share_I=1, share_C=0.6, share_D=0.6),
            "imaging-4": self.Distribution(share_I=1, share_C=0.2, share_D=1),
            "imaging-5": self.Distribution(share_I=1, share_C=0.2, share_D=1),
        }
        super().__init__(
            training, train_test_split_size, variety_acronym, labels_to_remove, distributions
        )
