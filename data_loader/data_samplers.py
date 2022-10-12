from typing import Protocol

import numpy as np
from sklearn.utils import shuffle


class BaseSampler(Protocol):
	def __init__(self, training):
		...

	def sample(self, images, labels, classes):
		...


class RandomSampler15(BaseSampler):
	def __init__(self, training):
		self.training = training

	def sample(self, images, labels, classes):
		idx_full = np.arange(len(images))

		np.random.seed(0)
		np.random.shuffle(idx_full)

		split = 0.15
		len_valid = int(len(images) * split)

		test_index = idx_full[0:len_valid]
		train_index = np.delete(idx_full, np.arange(0, len_valid))

		if self.training:
			images, classes = images[train_index], classes[train_index]
		else:
			images, classes = images[test_index], classes[test_index]
		images, classes = shuffle(images, classes, random_state=0)
		return images, classes




	# potato plant independant sampling
	# labels_test = [57, 64]
	# labels_test = [60, 64]
	# labels_train = [idx for idx in list(range(57,71)) if idx not in labels_test]
	# labels_train.pop(0)
	# train_index = np.concatenate([np.where(labels == label_) for label_ in labels_train], axis=1).ravel()
	# test_index = np.concatenate([np.where(labels == label_) for label_ in labels_test], axis=1).ravel()
