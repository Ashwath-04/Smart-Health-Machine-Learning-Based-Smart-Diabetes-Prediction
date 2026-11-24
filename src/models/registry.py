"""Candidate model registry: maps a candidate name to a factory with fixed settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.models.baselines import make_logistic_regression, make_random_forest
from src.models.catboost_model import make_catboost
from src.models.extratrees_model import make_extratrees
from src.models.stacking_model import make_soft_voting, make_stacking
from src.models.xgboost_model import make_xgboost

SINGLE_MODEL_FACTORIES: dict[str, Callable] = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "extratrees": make_extratrees,
    "catboost": make_catboost,
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
    "stacking": "Stacking (XGB+ET+CB -> LR)",
    "soft_voting": "Soft Voting (XGB+ET+CB)",
}


@dataclass
class CandidateSpec:
    """A fully specified experiment configuration."""

    family: str  # key of SINGLE_MODEL_FACTORIES, "stacking" or "soft_voting"
    smote: bool
    tuned: bool
    params: dict[str, Any] = field(default_factory=dict)  # single model params
    base_params: dict[str, dict] = field(default_factory=dict)  # ensembles
    meta_params: dict[str, Any] = field(default_factory=dict)  # stacking meta learner

    @property
    def name(self) -> str:
        return f"{self.family}_{'smote' if self.smote else 'nosmote'}_{'tuned' if self.tuned else 'default'}"

    @property
    def display_name(self) -> str:
        base = DISPLAY_NAMES.get(self.family, self.family)
        return f"{base} | {'SMOTE' if self.smote else 'no SMOTE'} | {'tuned' if self.tuned else 'default'}"

    def build(self, cfg, seed: int | None = None):
        if self.family == "stacking":
            return make_stacking(cfg, self.base_params, self.meta_params, smote=self.smote, seed=seed)
        if self.family == "soft_voting":
            return make_soft_voting(cfg, self.base_params, smote=self.smote, seed=seed)
        return SINGLE_MODEL_FACTORIES[self.family](cfg, self.params, smote=self.smote, seed=seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "smote": self.smote,
            "tuned": self.tuned,
            "params": self.params,
            "base_params": self.base_params,
            "meta_params": self.meta_params,
        }
