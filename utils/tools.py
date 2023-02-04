import numpy as np
from scipy.signal import find_peaks
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


def find_signal_peaks(vips):
    peak_indexes, properties = find_peaks(x=vips, height=1.5, distance=20)
    peak_heights = properties["peak_heights"]
    peak_heights_dict = {key: value for (key, value) in zip(peak_indexes, peak_heights)}
    return peak_heights_dict, peak_indexes
