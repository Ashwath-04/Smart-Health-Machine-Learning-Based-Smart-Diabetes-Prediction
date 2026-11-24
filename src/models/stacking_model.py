"""Stacking ensemble: XGBoost + ExtraTrees + CatBoost -> Logistic Regression.

How the meta-features are generated (sklearn ``StackingClassifier``):

1. For each base pipeline, ``cross_val_predict`` is run on the training data with
   ``cv=StratifiedKFold(inner_splits)``. Each base pipeline (impute -> SMOTE -> model) is
   fitted on k-1 inner folds and predicts ``P(y=1)`` for the held-out inner fold. SMOTE
   runs only on those k-1 folds, so every meta-feature is an *out-of-fold* prediction for
   a *real* (never synthetic) patient that the producing base model did not see.
2. The meta learner (Logistic Regression) is fitted on this n x 3 out-of-fold matrix
   against the true labels. Because the meta learner sees the real class balance, its
   output is re-anchored to the true prevalence (SMOTE shifts base-model probabilities).
3. Each base pipeline is then refitted on the full training data (``estimators_``); at
   inference time their probabilities are fed to the fitted meta learner.

For binary problems sklearn drops the redundant ``P(y=0)`` column, giving 3 meta-features.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.models.catboost_model import make_catboost
from src.models.extratrees_model import make_extratrees
from src.models.xgboost_model import make_xgboost

BASE_LEARNERS = ("xgboost", "extratrees", "catboost")
_BASE_FACTORIES = {
    "xgboost": make_xgboost,
    "extratrees": make_extratrees,
    "catboost": make_catboost,
}


def make_meta_learner(cfg, params: dict[str, Any] | None = None, seed=None) -> LogisticRegression:
    seed = cfg.seed if seed is None else seed
    p = {**cfg["models"]["meta_learner"], **(params or {})}
    return LogisticRegression(max_iter=5000, random_state=seed, **p)


def base_estimators(cfg, base_params: dict[str, dict] | None, smote: bool, seed=None) -> list[tuple[str, Any]]:
    base_params = base_params or {}
    return [
        (name, _BASE_FACTORIES[name](cfg, base_params.get(name), smote=smote, seed=seed))
        for name in BASE_LEARNERS
    ]


def make_stacking(
    cfg,
    base_params: dict[str, dict] | None = None,
    meta_params: dict[str, Any] | None = None,
    smote: bool = True,
    seed=None,
) -> StackingClassifier:
    """Build the stacking ensemble. Each base estimator carries its own preprocessing+SMOTE."""
    seed = cfg.seed if seed is None else seed
    return StackingClassifier(
        estimators=base_estimators(cfg, base_params, smote, seed),
        final_estimator=make_meta_learner(cfg, meta_params, seed),
        cv=StratifiedKFold(cfg["cv"]["stacking_inner_splits"], shuffle=True, random_state=seed),
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,
    )


def make_soft_voting(cfg, base_params: dict[str, dict] | None = None, smote: bool = True, seed=None):
    """Unweighted probability average of the same base learners (ablation for meta-learning)."""
    return VotingClassifier(estimators=base_estimators(cfg, base_params, smote, seed), voting="soft", n_jobs=1)
