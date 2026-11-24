"""ExtraTrees base learner and its Optuna search space."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier

from src.models.common import SMOTE_K_PARAM, build_pipeline, split_smote_params


def make_extratrees(cfg, params: dict[str, Any] | None = None, smote: bool = False, seed=None):
    seed = cfg.seed if seed is None else seed
    p, k = split_smote_params({**cfg["models"]["extratrees"], **(params or {})})
    clf = ExtraTreesClassifier(random_state=seed, n_jobs=1, **p)
    return build_pipeline(clf, cfg, smote=smote, smote_k_neighbors=k, seed=seed)


def suggest_extratrees_params(trial) -> dict[str, Any]:
    max_depth = trial.suggest_categorical("max_depth", [None, 4, 6, 8, 12, 16])
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": max_depth,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8, 1.0]),
        SMOTE_K_PARAM: trial.suggest_int(SMOTE_K_PARAM, 3, 10),
    }
