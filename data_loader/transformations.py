import random

import numpy as np
import torch


class AddGaussianNoise:
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
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class RandomCrop:
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


class RandomMirror:
    def __call__(self, image):
        # vertical = 0; horizontal = 1, no flip = None
        axis = random.choices([0, 1, (0, 1), None])[0]
        if axis is not None:
            image = np.flip(image, axis=axis)
        return image.astype(dtype="float32")


# x = np.random.rand(5, 5, 20).astype(dtype="float32")

# rand = RandomMirror()
# print(x.shape)
# print("")
# print(rand(x).shape)
