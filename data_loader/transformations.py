import random
from typing import Protocol

import numpy as np
import torch
from scipy.signal import savgol_filter


class Transform(Protocol):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        ...


class AddGaussianNoise(Transform):
    def __init__(self, mean=0.0, std=0.0):
        self.std = std
        self.mean = mean

    def __call__(self, image):
        noise_additive = np.random.uniform(low=-self.mean, high=+self.mean)
        noise_multiplicative = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise_additive + noise_multiplicative
        image[image < 0] = 0
        return image.astype(dtype="float32")

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class RandomCrop(Transform):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top : top + new_h, left : left + new_w]
        return image


class RandomMirror(Transform):
    def __call__(self, image):
        # vertical = 0; horizontal = 1, no flip = None
        axis = random.choices([0, 1, (0, 1), None])[0]
        if axis is not None:
            image = np.flip(image, axis=axis)
        return image.astype(dtype="float32")


class AreaNormalization(Transform):
    def __call__(self, image):
        image = self._image_normalization(image, self._signal_normalize)
        return image.astype(dtype="float32")

    @staticmethod
    def _image_normalization(image, func1d):
        return np.apply_along_axis(func1d, axis=2, arr=image)

    @staticmethod
    def _signal_normalize(signal):
        area = np.trapz(signal)
        if area == 0:
            return signal
        return signal / area


class SavgolTransform(Transform):
    WIN_LENGTH = 5
    POLYORDER = 2
    DERIV = 2

    def __call__(self, image):
        image = self._image_normalization(image, self._signal_normalize)
        return image.astype(dtype="float32")

    @staticmethod
    def _image_normalization(image, func1d):
        return np.apply_along_axis(func1d, axis=2, arr=image)

    def _signal_normalize(self, signal):
        if np.isnan(signal).any():
            return signal
        return savgol_filter(signal, self.WIN_LENGTH, self.POLYORDER, self.DERIV)


class NoTransformation(Transform):
    def __call__(self, image):
        return image
