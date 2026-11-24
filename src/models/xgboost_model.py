"""XGBoost base learner and its Optuna search space."""

from __future__ import annotations

from typing import Any

from xgboost import XGBClassifier

from src.models.common import SMOTE_K_PARAM, build_pipeline, split_smote_params


def make_xgboost(cfg, params: dict[str, Any] | None = None, smote: bool = False, seed=None):
    seed = cfg.seed if seed is None else seed
    p, k = split_smote_params({**cfg["models"]["xgboost"], **(params or {})})
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=1,  # parallelism happens across CV folds instead
        **p,
    )
    return build_pipeline(clf, cfg, smote=smote, smote_k_neighbors=k, seed=seed)


def suggest_xgboost_params(trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        SMOTE_K_PARAM: trial.suggest_int(SMOTE_K_PARAM, 3, 10),
    }
