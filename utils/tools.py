import numpy as np
from scipy.signal import find_peaks
from scipy.stats import bootstrap
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve


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


def calculate_roc_curve(data_df):
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    # bootstrapping used to get more smooth curve
    for s in range(1000):
        data_sampled = data_df.sample(n=len(data_df), replace=True, random_state=s)
        sample_pro = data_sampled["prediction_proba"].to_numpy()
        sample_label = data_sampled["target"].to_numpy()
        fpr_rf, tpr_rf, _ = roc_curve(sample_label, sample_pro)
        tprs.append(np.interp(mean_fpr, fpr_rf, tpr_rf))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    return mean_auc, mean_fpr, mean_tpr, tprs_upper, tprs_lower
