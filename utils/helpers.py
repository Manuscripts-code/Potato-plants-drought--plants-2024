import numpy as np
import torch
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def cross_val_model(model, X, y):
    model = clone(model)
    cross_val = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    result = cross_val_score(model, X, y,
                                cv=cross_val,
                                error_score=0,
                                n_jobs=1,
                                scoring='roc_auc')
    result_mean = result.mean()
    return round(result_mean, 2)

