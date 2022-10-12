import os

import numpy as np
import pandas as pd


class SPImage():
    def __init__(self, sp_file):
        self._sp_file = sp_file

    def __repr__(self):
        return repr(self._sp_file)

    def to_display(self, brightness):
        db = self.metadata["default bands"]
        db = list(map(int, db))
        image_3ch = self._sp_file.read_bands(db)
        image_3ch[:, :, 0] = image_3ch[:, :, 0] / (image_3ch[:, :, 0].max() / 255.) * brightness
        image_3ch[:, :, 1] = image_3ch[:, :, 1] / (image_3ch[:, :, 1].max() / 255.) * brightness
        image_3ch[:, :, 2] = image_3ch[:, :, 2] / (image_3ch[:, :, 2].max() / 255.) * brightness
        return image_3ch.astype("uint8")

    def to_numpy(self):
        return self._sp_file[:,:,:]

    def to_signatures(self, pixels_loc):
        image_arr = self.to_numpy()
        signatures = image_arr[pixels_loc.y, pixels_loc.x, :]
        data = {"x": pixels_loc.x, "y": pixels_loc.y, "signature": list(signatures)}
        return pd.DataFrame(data, columns=["x", "y", "signature"])

    def mean(self, axis=None):
        image_arr = self.to_numpy()
        return np.nanmean(image_arr, axis=axis)

    @property
    def file(self):
        return self._sp_file

    @property
    def metadata(self):
        return self._sp_file.metadata

    @property
    def shape(self):
        rows = self._sp_file.nrows
        samples = self._sp_file.ncols
        bands = self._sp_file.nbands
        return (rows, samples, bands)

    @property
    def rows(self):
        return self._sp_file.nrows

    @property
    def cols(self):
        return self._sp_file.ncols

    @property
    def bands(self):
        return self._sp_file.nbands

    @property
    def filename(self):
        return self._sp_file.filename.split(os.sep)[-1].split(".")[0]

    @property
    def label(self):
        path_split = self.filename.split("__")
        idx = int(path_split[0].split("_")[1])
        labels_all = path_split[1].split("_")
        return labels_all[idx-1]

    @property
    def wavelengths(self):
        wavelen = self._sp_file.metadata["wavelength"]
        return list(map(float, wavelen))
