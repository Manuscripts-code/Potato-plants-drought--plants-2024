import glob
from pathlib import Path

import numpy as np
import pandas as pd
import spectral as sp
from rich.progress import track
from torch.utils.data import Dataset
from torchvision import transforms

from .sp_image import SPImage
from .transformations import (AddGaussianNoise, AreaNormalization,
                              NoTransformation, RandomCrop, RandomMirror)

# hardcoded bands to remove
NOISY_BANDS = np.concatenate([np.arange(26), np.arange(140, 171), np.arange(430, 448)])


class PlantsDataset(Dataset):
    def __init__(self, data_dir, data_sampler, grouped_labels_filepath, training):
        self.train = training
        (
            self.transform_train,
            self.transform_test,
            self.transform_during_loading,
        ) = self._init_transform()

        images, labels, classes = self._read_data(data_dir, grouped_labels_filepath)
        # get data based on whether it is training or testing run
        self.images, self.classes = data_sampler(images, labels, classes)

    def _init_transform(self):
        transform_train = transforms.Compose(
            [
                RandomMirror(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_during_loading = transforms.Compose([AreaNormalization()])
        transform_during_loading = transforms.Compose([NoTransformation()])
        return transform_train, transform_test, transform_during_loading

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
        for path in track(images_paths, description="Loading images..."):
            image = SPImage(sp.envi.open(path))
            image_label = image.label

            if image_label not in groups.labels.values:
                continue

            image_group = groups[groups.labels == image_label].groups_encoded.iloc[0]

            # convert image to array
            image_arr = image.to_numpy()
            # remove noisy channels
            image_arr = np.delete(image_arr, NOISY_BANDS, axis=2)
            # clip between 0 and 1
            image_arr = image_arr.clip(0, 1)
            # transform by transformations defined during loading
            image_arr = self.transform_during_loading(image_arr)

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
