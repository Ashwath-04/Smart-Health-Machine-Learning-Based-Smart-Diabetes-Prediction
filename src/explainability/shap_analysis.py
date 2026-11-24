"""SHAP explanations of the deployed model's *calibrated probability output*.

The stacking ensemble is not a single tree model, so a model-agnostic explainer is used:
``shap.explainers.Exact`` over the 8 input features (2^8 coalitions, exact Shapley values
relative to a background sample of training patients). Explanations are computed in the
imputed feature space: impossible zeros / missing inputs are first replaced by the training
medians (exactly what the fitted pipeline does), so each SHAP value refers to a concrete
input value.

SHAP values describe how the *model* moved its predicted probability away from the average
prediction. They are model contributions, not clinical causes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.data.preprocessing import ZeroToNaN
from src.evaluation.plots import _save, plt
from src.models.bundle import ModelBundle


def impute_like_training(X: pd.DataFrame, bundle: ModelBundle, zero_as_missing: list[str]) -> pd.DataFrame:
    """Replace impossible zeros / NaN with the training medians stored in the bundle."""
    X = X[bundle.features].astype(float)
    X = ZeroToNaN(zero_as_missing).fit(X).transform(X)
    return X.fillna(bundle.imputation_medians)


def background_sample(X_train_imputed: pd.DataFrame, y_train: pd.Series | None, size: int, seed: int) -> pd.DataFrame:
    """Stratified random background sample (preserves class balance of the training data)."""
    if size >= len(X_train_imputed):
        return X_train_imputed
    if y_train is None:
        return X_train_imputed.sample(size, random_state=seed)
    from sklearn.model_selection import train_test_split

    bg, _ = train_test_split(X_train_imputed, train_size=size, stratify=y_train, random_state=seed)
    return bg


class ShapExplainer:
    """Wraps an exact SHAP explainer around ``bundle.predict_proba``."""

    def __init__(self, bundle: ModelBundle, background: pd.DataFrame):
        self.bundle = bundle
        self.features = list(bundle.features)
        self.background = background[self.features].astype(float)
        masker = shap.maskers.Independent(self.background.to_numpy(), max_samples=len(self.background))
        self._explainer = shap.explainers.Exact(self._f, masker, feature_names=self.features)

    def _f(self, X: np.ndarray) -> np.ndarray:
        return self.bundle.predict_proba(pd.DataFrame(X, columns=self.features))

    @property
    def expected_value(self) -> float:
        return float(np.mean(self._f(self.background.to_numpy())))

    def explain(self, X_imputed: pd.DataFrame) -> shap.Explanation:
        exp = self._explainer(X_imputed[self.features].to_numpy())
        exp.feature_names = self.features
        return exp


def contributions(exp_row: shap.Explanation, raw_row: dict | None = None, imputed: list[str] | None = None) -> list[dict]:
    """Sorted per-feature contributions for a single explanation row."""
    imputed = set(imputed or [])
    items = []
    for name, value, sv in zip(exp_row.feature_names, exp_row.data, exp_row.values):
        items.append({
            "feature": name,
            "value": float(value),
            "imputed": name in imputed,
            "shap_value": float(sv),
            "direction": "increases" if sv > 0 else "decreases" if sv < 0 else "neutral",
        })
    return sorted(items, key=lambda d: abs(d["shap_value"]), reverse=True)


def global_importance(exp: shap.Explanation) -> pd.DataFrame:
    vals = np.abs(exp.values)
    return (
        pd.DataFrame({"feature": exp.feature_names, "mean_abs_shap": vals.mean(axis=0),
                      "mean_shap": exp.values.mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def save_global_plots(exp: shap.Explanation, figures_dir: Path, prefix: str = "shap") -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(exp, show=False, max_display=8)
    plt.title("SHAP summary: contribution to predicted probability (model, not causation)", fontsize=10)
    _save(plt.gcf(), figures_dir / f"{prefix}_summary_beeswarm.png")
    plt.figure()
    shap.plots.bar(exp, show=False, max_display=8)
    plt.title("Global importance: mean |SHAP| (probability units)", fontsize=10)
    _save(plt.gcf(), figures_dir / f"{prefix}_bar.png")
    for feat in ["Glucose", "BMI", "Age"]:
        if feat in exp.feature_names:
            plt.figure()
            shap.plots.scatter(exp[:, feat], show=False, color=exp)
            _save(plt.gcf(), figures_dir / f"{prefix}_dependence_{feat}.png")


def save_waterfall(exp_row: shap.Explanation, path: Path, title: str) -> None:
    plt.figure()
    shap.plots.waterfall(exp_row, show=False, max_display=8)
    plt.title(title, fontsize=10)
    _save(plt.gcf(), path)
