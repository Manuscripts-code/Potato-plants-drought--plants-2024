import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
# sns.set_style("whitegrid", {'axes.grid' : False})

from configs import configs
from data_loader import data_samplers
from notebooks.helpers import create_dataframe_from_absolute_paths
from utils.utils import ensure_dir, write_txt

# name to save the bar plot
SAVE_NAME = "countplot_KS"

# General configs (filer before applying sampler)
IMAGINGS = [
    "imaging-1",
    "imaging-2",
    "imaging-3",
    "imaging-4",
    "imaging-5",
]
VARIETIES = ["KS"]

# samplers configs if used
SAMPLER = "KrkaStratifySampler"
SAMPLER = None
TRAINING = True
TRAIN_TEST_SPLIT = 0.2

if __name__ == "__main__":
    data_dir_base = Path(configs.DATA_DIR)
    # get filepaths from data_dir_base and its subdirectories
    filepaths = glob.glob(str(data_dir_base / "**/*.hdr"), recursive=True)
    # create dataframe filed with metadata of samples
    df = create_dataframe_from_absolute_paths(filepaths)
    # filter out samples with imaging not in IMAGINGS or
    df = df[df["imaging"].isin(IMAGINGS)].reset_index(drop=True)
    df = df[df["variety"].isin(VARIETIES)].reset_index(drop=True)
    # filer with provided sampler
    if SAMPLER is not None:
        sampler = getattr(data_samplers, SAMPLER)(TRAINING, TRAIN_TEST_SPLIT)
        indices = np.arange(len(df))
        indices_sampled, _, _, _ = sampler(
            indices, df["identifier"].to_numpy(), df["class"].to_numpy(), df["imaging"].to_numpy()
        )
        df = df.iloc[indices_sampled].reset_index(drop=True)

    save_bar_plots_dir = ensure_dir(configs.BASE_DIR / "saved/bar_plots")

    ax = sns.countplot(data=df, x="imaging", hue="treatment")
    ax.grid(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines[['right', 'top']].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container)
    save_path = save_bar_plots_dir / f"{SAVE_NAME}.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
