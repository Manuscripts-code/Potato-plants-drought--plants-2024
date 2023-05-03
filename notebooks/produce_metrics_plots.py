import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from configs import configs
from utils.utils import ensure_dir, write_txt

if __name__ == "__main__":
    data = {
        "AUC-ROC": [0.74, 0.74, 0.87, 0.82, 0.64, 0.87, 0.76, 0.67],
        "Error": [0.04, 0.04, 0.03, 0.05, 0.07, 0.04, 0.08, 0.08],
        "Dataset": [
            "Unbiased--stratify-split",
            "Unbiased--random-split",
            "Biased-treatment--stratify-split",
            "Biased-imaging--stratify-split",
            "Unbiased--stratify-split",
            "Unbiased--random-split",
            "Biased-treatment--stratify-split",
            "Biased-imaging--stratify-split",
        ],
        "Variety": [
            "KIS Krka",
            "KIS Krka",
            "KIS Krka",
            "KIS Krka",
            "KIS Savinja",
            "KIS Savinja",
            "KIS Savinja",
            "KIS Savinja",
        ],
    }
    df = pd.DataFrame(data)
    df = df.sort_values(["Dataset", "Variety"])

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="Variety", y="AUC-ROC", data=df, hue="Dataset", palette="CMRmap_r", alpha=0.6)
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=df["Error"], fmt="none", c="k")

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=12)
    ax.grid(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(1)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(title="Dataset", loc="lower right", framealpha=1)
    ax.get_legend().remove()
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC-ROC", fontsize=14)
    ax.set_xlabel("Variety", fontsize=14)

    save_bar_plots_dir = ensure_dir(configs.BASE_DIR / "saved/bar_plots")
    save_path = save_bar_plots_dir / "barplot_metrics.pdf"
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
