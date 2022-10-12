import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import spectral as sp
from rich.progress import track
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from . import data_samplers as module_samplers
from .sp_image import SPImage


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)


class PlantsDataset(Dataset):
    def __init__(self, data_dir, data_sampler, grouped_labels_filepath, training):
        self.train = training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

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
        groups["groups_encoded"] = groups["groups"].astype('category').cat.codes

        images = []
        classes = []
        labels = []
        for path in track(images_paths, description="Loading images..."):
            image = SPImage(sp.envi.open(path))
            image_label = image.label
            
            if image_label not in groups.labels.values:
                continue

            image_group = groups[groups.labels == image_label].groups_encoded.iloc[0]

            # convect image to array and replace nans with zeros
            # TODO: remove noisy channels
            images.append(np.nan_to_num(image.to_numpy()))
            labels.append(image_label)
            classes.append(image_group)

        return np.array(images), np.array(labels), np.array(classes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])
        target = self.classes[idx]
        return (img, target)


class HypDataLoader(BaseDataLoader):
    def __init__(self, data_dir, data_sampler, grouped_labels_filepath, batch_size,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # init smapler from data_samplers.py
        data_sampler = getattr(module_samplers, data_sampler)(training=training)
        # init dataset which loads images by given sampler
        self.dataset = PlantsDataset(data_dir, data_sampler, grouped_labels_filepath, training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


