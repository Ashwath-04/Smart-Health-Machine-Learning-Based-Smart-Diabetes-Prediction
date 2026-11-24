"""Baseline models: Logistic Regression (scaled) and Random Forest (unscaled)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.common import build_pipeline, split_smote_params


def make_logistic_regression(cfg, params: dict[str, Any] | None = None, smote: bool = False, seed=None):
    """Logistic Regression is scale-sensitive, so this is the one pipeline with StandardScaler."""
    seed = cfg.seed if seed is None else seed
    p, k = split_smote_params({**cfg["models"]["logistic_regression"], **(params or {})})
    clf = LogisticRegression(random_state=seed, **p)
    return build_pipeline(clf, cfg, smote=smote, scale=True, smote_k_neighbors=k, seed=seed)


def make_random_forest(cfg, params: dict[str, Any] | None = None, smote: bool = False, seed=None):
    seed = cfg.seed if seed is None else seed
    p, k = split_smote_params({**cfg["models"]["random_forest"], **(params or {})})
    clf = RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    return build_pipeline(clf, cfg, smote=smote, scale=False, smote_k_neighbors=k, seed=seed)
