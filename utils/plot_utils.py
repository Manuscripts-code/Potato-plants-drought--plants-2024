import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from configs import configs
from utils.tools import find_signal_peaks


def plot_relavant_features(relevances):
    y = np.zeros_like(configs.BANDS_ORIGINAL)
    # remove the pre-defined noisy bands and assign values to the remaining places
    y[np.delete(np.arange(len(configs.BANDS_ORIGINAL)), configs.NOISY_BANDS)] = relevances
    # first append pre-defined noisy indices at the end (least relevant features)
    indices_by_relevance = np.argsort(y)[::-1]
    max_features = len(indices_by_relevance)

    features_rang = np.full(len(indices_by_relevance), 0)
    for idx, feature in enumerate(indices_by_relevance):
        features_rang[feature] = max_features - idx

    gradient = np.vstack((features_rang, features_rang))
    fig, ax = plt.subplots(nrows=1, figsize=(13, 1))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    # ax.set_title("Relavant features", fontsize=14)

    w_start, w_stop = configs.BANDS_ORIGINAL[0], configs.BANDS_ORIGINAL[-1]

    cmap = cm.get_cmap("viridis", 8)
    cmap.set_under("white")
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[w_start, w_stop, 1, 0], vmin=1)

    ax.set_xlabel("Wavelength [nm]")
    ax.xaxis.label.set_size(12)
    ax.set_yticklabels([])


def plot_relevances_amplitudes(relevances):
    y = np.zeros_like(configs.BANDS_ORIGINAL)
    # remove the pre-defined noisy bands and assign values to the remaining places
    y[np.delete(np.arange(len(configs.BANDS_ORIGINAL)), configs.NOISY_BANDS)] = relevances
    x = np.array(configs.BANDS_ORIGINAL)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 8))
    plt.plot(x, y, c="gray", linewidth=0.3, zorder=1)
    plt.scatter(x, y, c=y, s=3, cmap="viridis", zorder=2)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Wavelength [nm]")
    ax.xaxis.label.set_size(12)
    ax.set_ylabel("Relevance")
    ax.yaxis.label.set_size(12)
    peak_heights_dict, peak_indexes = find_signal_peaks(y)
    [plt.text(x[idx], peak_heights_dict[idx], int(x[idx])) for idx in peak_indexes]
    # plt.axhline(y=1.5, color="r", linestyle="-")
