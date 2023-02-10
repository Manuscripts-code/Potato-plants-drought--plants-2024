import glob
from pathlib import Path

import numpy as np
import spectral as sp
from rich.progress import track
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from configs import configs
from utils.utils import ensure_dir, save_image

from .sp_image import SPImage


class PlantsDataset(Dataset):
    def __init__(self, data_dir, imagings_used, data_sampler, training):
        self.imagings_used = imagings_used
        self.train = training
        self.transform_train = configs.TRANSFORM_TRAIN
        self.transform_test = configs.TRANSFORM_TEST
        self.transform_during_loading = configs.TRANSFORM_DURING_LOADING

        self.label_encoder = LabelEncoder()

        data_dir, self.use_cashed_images = self._get_data_dir(data_dir)
        images_paths = self._get_images_paths(data_dir)
        images, labels, classes, imagings = self._read_data(images_paths)
        # get data based on whether it is training or testing run
        self.images, self.labels, self.classes, self.imagings = data_sampler(
            images, labels, classes, imagings
        )
        self.classes = self.label_encoder.fit_transform(self.classes)

    def _get_data_dir(self, data_dir):
        data_dir = glob.glob(data_dir + "/*")
        if configs.USE_CASHED_IMAGES and configs.CASHED_IMAGES_DIR:
            data_cashed_dir = ensure_dir(configs.CASHED_IMAGES_DIR)
            if glob.glob(str(Path(configs.CASHED_IMAGES_DIR) / "*.hdr")):
                return data_cashed_dir, True
        return data_dir, False

    def _get_images_paths(self, data_dir):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]

        images_paths = []
        for data_dir_ in data_dir:
            images_paths += sorted(glob.glob(str(Path(data_dir_) / "*.hdr")))

        return images_paths

    def _read_data(self, images_paths):
        images = []
        classes = []
        labels = []
        imagings = []

        for path in track(images_paths, description="Loading images..."):
            image = SPImage(sp.envi.open(path))
            image_arr = self._prepare_image_arr(image)
            image_label = image.label
            imaging_label = self._extract_imaging_label(image.filename)
            image_group = self._map_label_to_group(image_label)

            if imaging_label in self.imagings_used:
                images.append(image_arr)
                labels.append(image_label)
                classes.append(image_group)
                imagings.append(imaging_label)

        return np.array(images), np.array(labels), np.array(classes), np.array(imagings)

    def _prepare_image_arr(self, image):
        # convert image to array
        image_arr = image.to_numpy()
        if self.use_cashed_images:
            return image_arr

        # replace nans with zeros
        image_arr = np.nan_to_num(image_arr)
        # clip between 0 and 1
        image_arr = image_arr.clip(0, 1)
        # transform by transformations defined during loading
        image_arr = self.transform_during_loading(image_arr)
        # remove noisy channels
        image_arr = np.delete(image_arr, configs.NOISY_BANDS, axis=2)

        if configs.SAVE_CASHED_IMAGES:
            save_image(
                configs.CASHED_IMAGES_DIR / (image.filename + ".hdr"),
                image_arr,
                metadata=image.metadata,
            )

        return image_arr

    @staticmethod
    def _extract_imaging_label(filename):
        return filename.split("__")[2]

    @staticmethod
    def _map_label_to_group(label):
        # remove number from label
        label = "-".join(label.split("-")[:2])
        if label in configs.GROUPS:
            group = configs.GROUPS[label]
        else:
            group = "unknown"
        return group

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.train:
            transform = self.transform_train
        else:
            transform = self.transform_test

        image = transform(self.images[idx])
        target = self.classes[idx]
        return (image, target, {"label": self.labels[idx], "imaging": self.imagings[idx]})
