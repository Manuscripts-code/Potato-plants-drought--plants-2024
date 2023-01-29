import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.cross_decomposition import PLSRegression


class PLSRegressionWrapper(PLSRegression):
    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)


class SavgolWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, win_length=7, polyorder=2, deriv=2):
        self.win_length = win_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        signatures_sav = []
        sp = [self.win_length, self.polyorder, self.deriv]
        for signal in X:
            if self.win_length != 0:
                signal = savgol_filter(signal, sp[0], sp[1], sp[2])
            signatures_sav.append(signal)
        return np.array(signatures_sav)


METHODS = {"SVC": SVC, "XGB": XGBClassifier, "savgol": SavgolWrapper, "PLS": PLSRegressionWrapper}
