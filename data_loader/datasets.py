import glob
from pathlib import Path

import numpy as np
import pandas as pd
import spectral as sp
from rich.progress import track
from torch.utils.data import Dataset
from torchvision import transforms

from .sp_image import SPImage
from .transformations import (AddGaussianNoise, AreaNormalization, RandomCrop,
                              RandomMirror)


class PlantsDataset(Dataset):
    def __init__(self, data_dir, data_sampler, grouped_labels_filepath, training):
        self.train = training
        self.transform_train = transforms.Compose(
            [
                # AddGaussianNoise(0.001, 0.002),
                RandomMirror(),
                # AreaNormalization(),
                transforms.ToTensor(),
            ]
        )
        self.transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # hardcoded bands to remove
        self.NOISY_BANDS = np.concatenate(
            [np.arange(26), np.arange(140, 171), np.arange(430, 448)]
        )

        images, labels, classes = self._read_data(data_dir, grouped_labels_filepath)
        # get data based on whether it is training or testing run
        self.images, self.classes = data_sampler.sample(images, labels, classes)

    def _read_data(self, data_dir, grouped_labels_filepath):
        data_dir_path = Path(data_dir)
        # read paths of hyperspectral images
        images_paths = sorted(glob.glob(str(data_dir_path / "*.hdr")))
        # read groups of labels
        groups = pd.read_excel(data_dir_path.parent / grouped_labels_filepath)
        # encode groups
        groups["groups_encoded"] = groups["groups"].astype("category").cat.codes

        images = []
        classes = []
        labels = []
        area_normalize = AreaNormalization()
        for path in track(images_paths, description="Loading images..."):
            image = SPImage(sp.envi.open(path))
            image_label = image.label

            if image_label not in groups.labels.values:
                continue

            image_group = groups[groups.labels == image_label].groups_encoded.iloc[0]

            # convect image to array and replace nans with zeros
            image_arr = np.nan_to_num(image.to_numpy())
            # normalize by area under the signal
            image_arr = area_normalize(image_arr)
            # and remove noisy channels
            image_arr = np.delete(image_arr, self.NOISY_BANDS, axis=2)

            images.append(image_arr)
            labels.append(image_label)
            classes.append(image_group)

        return np.array(images), np.array(labels), np.array(classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.train:
            transform = self.transform_train
        else:
            transform = self.transform_test

        img = transform(self.images[idx])
        target = self.classes[idx]
        return (img, target)
