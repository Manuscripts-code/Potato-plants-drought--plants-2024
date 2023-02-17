import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from configs import configs
from utils.tools import calculate_roc_curve, find_signal_peaks


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


def plot_roc_curves(data_dfs, title=""):
    if not isinstance(data_dfs, list):
        data_dfs = [data_dfs]

    cmap = plt.get_cmap("viridis")
    no_colors = len(data_dfs)
    colors = cmap(np.linspace(0, 1, no_colors))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance (AUC=0.5)", alpha=0.9)
    for idx, data_df in enumerate(data_dfs):
        mean_auc, mean_fpr, mean_tpr, tprs_upper, tprs_lower = calculate_roc_curve(data_df)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=colors[idx],
            label=f"data {idx} (AUC={mean_auc:.2f})",
            lw=2,
            alpha=0.9,
        )
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[idx], alpha=0.5)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.tick_params(labelsize=22)
    ax.set_title(title, fontsize=24)
    ax.set_ylabel("Sensitivity", fontsize=24)
    ax.set_xlabel("1-Specificity", fontsize=24)
    ax.legend(loc="lower right", fontsize=22)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    # plt.savefig("AUC.pdf", format="pdf")
