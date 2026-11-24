"""Repeated stratified cross-validation with out-of-fold (OOF) predictions.

Every fold clones the *entire* pipeline (impute -> SMOTE -> model, or the full stack), fits
it on the training fold only and predicts the untouched validation fold. Validation rows
are never resampled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold

from src.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    fold_metrics: pd.DataFrame  # one row per (repeat, fold), metrics at threshold 0.5
    oof: pd.DataFrame  # columns: repeat, fold, row_id, y_true, proba
    fit_seconds: float


def _fit_predict_fold(estimator, X: pd.DataFrame, y: pd.Series, tr: np.ndarray, va: np.ndarray):
    model = clone(estimator)
    model.fit(X.iloc[tr], y.iloc[tr])
    return model.predict_proba(X.iloc[va])[:, 1]


def cross_validate_model(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    n_jobs: int = -1,
) -> CVResult:
    """Run repeated stratified k-fold CV and collect per-fold metrics and OOF probabilities."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    splits = list(rskf.split(X, y))
    start = time.perf_counter()
    probas = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict_fold)(estimator, X, y, tr, va) for tr, va in splits
    )
    elapsed = time.perf_counter() - start

    fold_rows, oof_parts = [], []
    for i, ((_, va), p) in enumerate(zip(splits, probas)):
        repeat, fold = divmod(i, n_splits)
        y_va = y.iloc[va].to_numpy()
        fold_rows.append({"repeat": repeat, "fold": fold, **compute_metrics(y_va, p, 0.5)})
        oof_parts.append(
            pd.DataFrame(
                {"repeat": repeat, "fold": fold, "row_id": X.index[va], "y_true": y_va, "proba": p}
            )
        )
    return CVResult(pd.DataFrame(fold_rows), pd.concat(oof_parts, ignore_index=True), elapsed)
