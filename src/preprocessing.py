"""Leakage-safe preprocessing components for AQI modeling."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IQRCapper(BaseEstimator, TransformerMixin):
    """Cap each feature to training-set Tukey fences."""
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        spread = q3 - q1
        self.lower_ = q1 - self.factor * spread
        self.upper_ = q3 + self.factor * spread
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def make_preprocessor() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("outlier_capper", IQRCapper()), ("scaler", StandardScaler())])


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
