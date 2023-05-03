import pandas as pd

from configs import configs
from notebooks.helpers import create_per_imaging_report, get_plot_name, load_test_df
from utils.utils import ensure_dir, write_txt

if __name__ == "__main__":
    # define mlflow run indexes
    run_ids = ["6820e4edf73f4655827f8aeefe659e54", "109c91055eca4d4684f54b4636629821"]
    test_data = [(load_test_df(run_id, training=False)) for run_id in run_ids]

    save_metrics_report_dir = ensure_dir(configs.BASE_DIR / "saved/metrics_report")
    report = ""
    for test_df, config in test_data:
        name = get_plot_name(config)

        # remove irrelevant columns
        test_df = test_df[["imaging", "label", "target", "prediction"]]

        targets = test_df.groupby(["imaging", "label"]).max()["target"]
        predictions = test_df.groupby(["imaging", "label"]).sum()["prediction"]
        size = test_df.groupby(["imaging", "label"]).size()

        test_df = pd.concat([size, targets, predictions], axis=1)
        test_df["prediction_proba"] = test_df["prediction"] / test_df[0]

        test_df["prediction"][test_df["prediction_proba"] >= 0.5] = 1
        test_df["prediction"][test_df["prediction_proba"] < 0.5] = 0

        report += f"{name}\n"
        report += create_per_imaging_report(test_df, add_counts=False)

    write_txt(report, save_metrics_report_dir / "metrics_report_voting.txt")
