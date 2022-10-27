import sklearn.metrics as metrics_
from sklearn.model_selection import cross_val_score

AVAILABLE_SCORING_METRICS = {
    "accuracy": "accuracy_score",
    "balanced_accuracy": "balanced_accuracy_score",
}


def make_scorer_ftn(scoring_metric, init=True):
    if scoring_metric not in AVAILABLE_SCORING_METRICS.keys():
        raise Exception(
            f"Scoring metric {scoring_metric} not available. Possible options: {list(AVAILABLE_SCORING_METRICS.keys())}"
        )
    if not init:
        return scoring_metric

    scoring_metric_ftn = getattr(metrics_, AVAILABLE_SCORING_METRICS[scoring_metric])
    return scoring_metric_ftn


def objective_cv(model, X_data, y_data, validator, scoring_metric):
    score = cross_val_score(
        model,
        X_data,
        y_data,
        cv=validator,
        error_score=0,
        n_jobs=1,
        pre_dispatch=1,
        scoring=scoring_metric,
    )
    return score.mean()


def objective_split(model, X_train, y_train, X_valid, y_valid, scoring_metric_ftn):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    score = scoring_metric_ftn(y_valid, y_pred)
    return score
