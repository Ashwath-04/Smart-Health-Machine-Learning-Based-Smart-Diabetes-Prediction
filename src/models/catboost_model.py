"""CatBoost base learner and its Optuna search space."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier

from src.models.common import SMOTE_K_PARAM, build_pipeline, split_smote_params


def make_catboost(cfg, params: dict[str, Any] | None = None, smote: bool = False, seed=None):
    seed = cfg.seed if seed is None else seed
    p, k = split_smote_params({**cfg["models"]["catboost"], **(params or {})})
    clf = CatBoostClassifier(
        random_seed=seed,
        verbose=0,
        thread_count=1,  # parallelism happens across CV folds instead
        allow_writing_files=False,
        **p,
    )
    return build_pipeline(clf, cfg, smote=smote, smote_k_neighbors=k, seed=seed)


def suggest_catboost_params(trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 800, step=50),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        SMOTE_K_PARAM: trial.suggest_int(SMOTE_K_PARAM, 3, 10),
    }
