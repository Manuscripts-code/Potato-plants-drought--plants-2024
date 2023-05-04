import matplotlib.pyplot as plt
import numpy as np

from configs import configs
from notebooks.helpers import (
    create_per_imaging_report,
    create_wavelengths_report,
    get_plot_name,
    load_ids_from_registry,
    load_test_df,
)
from utils.plot_utils import (
    plot_relavant_features,
    plot_relevances_amplitudes,
    plot_roc_curves,
    plot_signatures,
)
from utils.utils import ensure_dir, write_json, write_txt

TRAINING = False


def produce_results(run_id=None):
    if run_id is not None:
        run_ids = [run_id]
    else:
        # load test_data
        run_ids = load_ids_from_registry()
        # run_ids = ["6820e4edf73f4655827f8aeefe659e54", "109c91055eca4d4684f54b4636629821"]

    test_data = [(load_test_df(run_id, training=TRAINING)) for run_id in run_ids]

    # define directories where the outputs will be saved
    save_roc_curves_dir = ensure_dir(configs.BASE_DIR / "saved/roc_curves")
    save_metrics_report_dir = ensure_dir(configs.BASE_DIR / "saved/metrics_report")
    save_relevances_plot_dir = ensure_dir(configs.BASE_DIR / "saved/relevances_plot")
    save_relevances_bar_dir = ensure_dir(configs.BASE_DIR / "saved/relevances_bar")
    save_wavelengths_dir = ensure_dir(configs.BASE_DIR / "saved/wavelengths")
    save_signatures_dir = ensure_dir(configs.BASE_DIR / "saved/signatures")
    report = ""

    for test_df, config in test_data:
        name = get_plot_name(config)

        # roc curves
        dfs = [df for _, df in test_df.groupby("imaging")]
        plot_roc_curves(dfs, title=name)
        # save_path = save_roc_curves_dir / f"{name}.png"
        # plt.savefig(save_path)
        save_path = save_roc_curves_dir / f"{name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

        # relevances
        relevances = np.mean(test_df["relevance"], axis=0)
        plot_relevances_amplitudes(relevances, title=name)
        # save_path = save_relevances_plot_dir / f"{name}.png"
        # plt.savefig(save_path)
        save_path = save_relevances_plot_dir / f"{name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

        plot_relavant_features(relevances)
        # save_path = save_relevances_bar_dir / f"{name}.png"
        # plt.savefig(save_path)
        save_path = save_relevances_bar_dir / f"{name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

        # save relevant wavelengths
        wavelengths_report = create_wavelengths_report(relevances)
        write_json(wavelengths_report, save_wavelengths_dir / f"{name}.json")

        # signatures
        plot_signatures(test_df["signature"].to_numpy(), test_df["target"].to_numpy(), title=name)
        # save_path = save_signatures_dir / f"{name}.png"
        # plt.savefig(save_path)
        save_path = save_signatures_dir / f"{name}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight")

        # metrics report
        report += f"{name}\n"
        report += create_per_imaging_report(test_df, add_counts=False)

    write_txt(report, save_metrics_report_dir / "metrics_report.txt")


if __name__ == "__main__":
    produce_results()
