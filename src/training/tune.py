"""Optuna hyperparameter optimisation for the base learners and the meta learner.

Each Optuna trial evaluates the *whole* pipeline (impute -> SMOTE -> model) with stratified
CV on the training set only. The test set is never touched. SMOTE's ``k_neighbors`` is tuned
jointly with the model hyperparameters.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.models.catboost_model import make_catboost, suggest_catboost_params
from src.models.extratrees_model import make_extratrees, suggest_extratrees_params
from src.models.stacking_model import BASE_LEARNERS
from src.models.xgboost_model import make_xgboost, suggest_xgboost_params

logger = logging.getLogger(__name__)

SEARCH_SPACES: dict[str, tuple[Callable, Callable]] = {
    "xgboost": (make_xgboost, suggest_xgboost_params),
    "extratrees": (make_extratrees, suggest_extratrees_params),
    "catboost": (make_catboost, suggest_catboost_params),
}


def tune_base_learner(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cfg,
    n_trials: int,
    smote: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune one base learner; returns (best params, trials dataframe)."""
    factory, suggest = SEARCH_SPACES[name]
    tcfg = cfg["tuning"]
    cv = StratifiedKFold(tcfg["n_splits"], shuffle=True, random_state=tcfg["cv_seed"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        model = factory(cfg, params, smote=smote)
        scores = cross_val_score(model, X, y, cv=cv, scoring=tcfg["metric"], n_jobs=cfg["cv"]["n_jobs"])
        trial.set_user_attr("cv_std", float(np.std(scores)))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        study_name=f"{name}_tuning",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("%s best CV %s = %.4f", name, tcfg["metric"], study.best_value)
    return dict(study.best_params), study.trials_dataframe()


def base_oof_meta_features(
    base_params: dict[str, dict], X: pd.DataFrame, y: pd.Series, cfg, smote: bool = True
) -> pd.DataFrame:
    """OOF ``P(y=1)`` of each base pipeline: the exact meta-features StackingClassifier builds."""
    cv = StratifiedKFold(cfg["cv"]["stacking_inner_splits"], shuffle=True, random_state=cfg.seed)
    factories = {n: SEARCH_SPACES[n][0] for n in BASE_LEARNERS}
    cols = {}
    for name in BASE_LEARNERS:
        model = factories[name](cfg, base_params.get(name), smote=smote)
        cols[name] = cross_val_predict(
            model, X, y, cv=cv, method="predict_proba", n_jobs=cfg["cv"]["n_jobs"]
        )[:, 1]
    return pd.DataFrame(cols, index=X.index)


def tune_meta_learner(Z: pd.DataFrame, y: pd.Series, cfg, n_trials: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Tune the Logistic Regression meta learner on the OOF meta-feature matrix ``Z``."""
    tcfg = cfg["tuning"]
    cv = StratifiedKFold(tcfg["n_splits"], shuffle=True, random_state=tcfg["cv_seed"])

    def objective(trial: optuna.Trial) -> float:
        params = suggest_meta_params(trial)
        model = LogisticRegression(max_iter=5000, **params)
        return float(np.mean(cross_val_score(model, Z, y, cv=cv, scoring=tcfg["metric"])))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = suggest_meta_params(optuna.trial.FixedTrial(study.best_params))
    return best, study.trials_dataframe()


def suggest_meta_params(trial) -> dict[str, Any]:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    return {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        # sklearn>=1.8 expresses the penalty through l1_ratio (1 = L1, 0 = L2).
        "l1_ratio": 1.0 if penalty == "l1" else 0.0,
        "solver": "liblinear" if penalty == "l1" else "lbfgs",
    }
