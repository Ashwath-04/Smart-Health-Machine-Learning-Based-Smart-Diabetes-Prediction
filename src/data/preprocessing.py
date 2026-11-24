"""Leakage-safe preprocessing.

Design decisions (documented in data/README.md):

* Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are physiologically
  impossible for a living adult and are converted to NaN. Zeros in Pregnancies are genuine.
* Missing values are imputed with the *training-fold* median (``SimpleImputer`` is fitted
  inside every CV fold / pipeline, never on the full dataset).
* Scaling is only added for models that need it (Logistic Regression). Tree ensembles get
  the unscaled imputed features.
* Every transformer is a scikit-learn estimator so the whole chain can be cloned and
  refitted per fold by ``cross_validate``/``StackingClassifier``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineering import ClinicalFeatureEngineer


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace physiologically impossible zeros with NaN in the given columns.

    Stateless: ``fit`` only records column names, so it cannot leak information.
    """

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        X = self._as_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        missing = set(self.columns or []) - set(X.columns)
        if missing:
            raise ValueError(f"ZeroToNaN columns not in input: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._as_frame(X).copy()
        for col in self.columns or []:
            X[col] = X[col].astype(float).mask(X[col] == 0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_

    def _as_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        names = getattr(self, "feature_names_in_", None)
        return pd.DataFrame(X, columns=names)


def preprocessing_steps(
    zero_as_missing: list[str],
    scale: bool = False,
    add_missing_indicators: bool = False,
    engineered_features: bool = False,
) -> list[tuple[str, object]]:
    """Per-model preprocessing steps, each emitting a pandas DataFrame.

    Returned as a flat list because imblearn pipelines do not accept a nested ``Pipeline``
    as an intermediate step.
    """
    steps: list[tuple[str, object]] = [
        ("zero_to_nan", ZeroToNaN(columns=list(zero_as_missing))),
        ("imputer", SimpleImputer(strategy="median", add_indicator=add_missing_indicators)),
    ]
    if engineered_features:
        steps.append(("features", ClinicalFeatureEngineer()))
    if scale:
        steps.append(("scaler", StandardScaler()))
    for _, step in steps:
        step.set_output(transform="pandas")
    return steps


def build_preprocessor(zero_as_missing: list[str], **kwargs) -> Pipeline:
    """Standalone preprocessing ``Pipeline`` (used for analysis and tests)."""
    return Pipeline(preprocessing_steps(zero_as_missing, **kwargs))


def preprocessing_steps_from_config(cfg, scale: bool = False) -> list[tuple[str, object]]:
    pp = cfg["preprocessing"]
    if pp.get("imputation", "median") != "median":
        raise ValueError("Only median imputation is supported in this version.")
    return preprocessing_steps(
        zero_as_missing=cfg["data"]["zero_as_missing"],
        scale=scale,
        add_missing_indicators=bool(pp.get("add_missing_indicators", False)),
        engineered_features=bool(pp.get("engineered_features", False)),
    )


def training_medians(X_train: pd.DataFrame, zero_as_missing: list[str]) -> dict[str, float]:
    """Medians used by the imputer when fitted on the full training set (for metadata/UI)."""
    cleaned = ZeroToNaN(zero_as_missing).fit(X_train).transform(X_train)
    return {c: float(cleaned[c].median()) for c in cleaned.columns}
