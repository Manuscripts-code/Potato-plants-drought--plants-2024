import numpy as np
from scipy.signal import find_peaks
from scipy.stats import bootstrap
from sklearn.metrics import precision_recall_fscore_support


def calculate_classification_metrics(y_true, y_pred, classes=None):
    """Performance metrics using ground truths and predictions.

    Args:
        y_true (np.ndarray): true labels.
        y_pred (np.ndarray): predicted labels.
        classes (List[str]): list of class labels.

    Returns:
        Dict: performance metrics.
    """
    # Performance
    metrics = {"overall": {}, "class": {}}

    # Overall metrics
    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    metrics["overall"]["precision"] = overall_metrics[0]
    metrics["overall"]["recall"] = overall_metrics[1]
    metrics["overall"]["f1"] = overall_metrics[2]
    metrics["overall"]["num_samples"] = np.float64(len(y_true))

    # Per-class metrics
    if classes is None:
        classes = np.unique(y_true).astype("str")

    class_metrics = precision_recall_fscore_support(y_true, y_pred, average=None)
    for i, _class in enumerate(classes):
        metrics["class"][_class] = {
            "precision": class_metrics[0][i],
            "recall": class_metrics[1][i],
            "f1": class_metrics[2][i],
            "num_samples": np.float64(class_metrics[3][i]),
        }

    return metrics


def find_signal_peaks(signal):
    peak_indexes, properties = find_peaks(x=signal, height=0.5, distance=20)
    peak_heights = properties["peak_heights"]
    peak_heights_dict = {key: value for (key, value) in zip(peak_indexes, peak_heights)}
    return peak_heights_dict, peak_indexes


def calculate_confidence_interval(data, statistic):
    res = bootstrap(
        data,
        statistic=statistic,
        n_resamples=1000,
        confidence_level=0.95,
        random_state=0,
        paired=True,
        vectorized=False,
        method="BCa",
    )
    return res.confidence_interval


def calculate_metric_and_confidence_interval(data_df, metric):
    data = (data_df["target"].to_numpy(), data_df["prediction"].to_numpy())
    mean = metric(*data)
    ci = calculate_confidence_interval(data, statistic=metric)
    return mean, ci
