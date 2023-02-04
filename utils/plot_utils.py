import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from configs import configs
from utils.tools import find_signal_peaks


def plot_color_gradients(features_rang):
    gradient = np.vstack((features_rang, features_rang))
    fig, ax = plt.subplots(nrows=1, figsize=(13, 1))
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    # ax.set_title("Relavant features", fontsize=14)

    w_start, w_stop = configs.BANDS[0], configs.BANDS[-1]

    cmap = cm.get_cmap("viridis", 8)
    cmap.set_under("white")
    ax.imshow(gradient, aspect="auto", cmap=cmap, extent=[w_start, w_stop, 1, 0], vmin=1)

    ax.set_xlabel("Wavelength [nm]")
    ax.xaxis.label.set_size(12)
    ax.set_yticklabels([])


def plot_relavant_features(bands_relavance, max_features=None):
    """Plot relavant features in bins by wavelength

    Args:
                bands_relavance ([list]): relavance of bands from most relavant to least
                max_features (int): maximum number of most relavant features to plot
    """
    if max_features is None or max_features > len(bands_relavance):
        max_features = len(bands_relavance)

    features_rang = np.full(len(bands_relavance), 0)
    for idx, feature in enumerate(bands_relavance):
        if max_features == idx:
            break
        features_rang[feature] = max_features - idx

    plot_color_gradients(features_rang)


def plot_relevances_amplitudes(relevances):
    x = np.array(configs.BANDS)
    points = np.array([x, relevances]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(10, 8))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(relevances.min(), relevances.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm)
    # Set the values used for colormapping
    lc.set_array(relevances)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("Wavelength [nm]")
    ax.xaxis.label.set_size(12)
    ax.set_ylabel("Relevance")
    ax.yaxis.label.set_size(12)
    peak_heights_dict, peak_indexes = find_signal_peaks(relevances)
    [plt.text(x[idx], peak_heights_dict[idx], int(x[idx])) for idx in peak_indexes]
    plt.axhline(y=1.5, color="r", linestyle="-")
	