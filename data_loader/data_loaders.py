import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

from . import data_samplers as module_samplers
from . import datasets as module_datasets


class BaseDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=default_collate,
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
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
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
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


class HypDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        dataset,
        data_sampler,
        grouped_labels_filepath,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        num_workers=1,
        training=True,
    ):
        # init smapler from data_samplers.py
        data_sampler = getattr(module_samplers, data_sampler)(training=training)
        # init dataset which loads images by given sampler
        self.dataset = getattr(module_datasets, dataset)(
            data_dir, data_sampler, grouped_labels_filepath, training
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
