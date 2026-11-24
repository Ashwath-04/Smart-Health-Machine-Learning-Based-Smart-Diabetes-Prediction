"""Optional engineered features.

Disabled by default (``preprocessing.engineered_features: false``): the tree ensembles can
learn these interactions themselves, and with 614 training rows every extra feature adds
variance. The transformer exists so the choice can be tested by flipping one config flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add simple, stateless interaction features after imputation."""

    new_features = ["Glucose_x_BMI", "Insulin_to_Glucose"]

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()
        X["Glucose_x_BMI"] = X["Glucose"] * X["BMI"] / 1000.0
        X["Insulin_to_Glucose"] = X["Insulin"] / X["Glucose"].clip(lower=1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.new_features, dtype=object)
