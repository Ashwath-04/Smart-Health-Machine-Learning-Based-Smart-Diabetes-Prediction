"""Complementary importance views: permutation importance and native base-learner importances."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.plots import PALETTE, _save, plt
from src.models.bundle import ModelBundle


def permutation_importance_ap(bundle: ModelBundle, X: pd.DataFrame, y, n_repeats: int = 20, seed: int = 42) -> pd.DataFrame:
    """Drop in PR-AUC when one feature column is shuffled (model-agnostic, on held-out data)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    base = average_precision_score(y, bundle.predict_proba(X))
    rows = []
    for f in bundle.features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].to_numpy())
            drops.append(base - average_precision_score(y, bundle.predict_proba(Xp)))
        rows.append({"feature": f, "pr_auc_drop_mean": float(np.mean(drops)), "pr_auc_drop_std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True)


def native_importances(bundle: ModelBundle) -> pd.DataFrame:
    """Built-in importances of each fitted base learner inside a stacking/voting model,
    normalised to sum to 1 per learner (gain for XGBoost, MDI for ExtraTrees,
    PredictionValuesChange for CatBoost)."""
    model = bundle.model
    estimators = getattr(model, "named_estimators_", None)
    if estimators is None:
        estimators = {"model": model}
    out = {}
    for name, pipe in estimators.items():
        clf = pipe.named_steps["clf"] if hasattr(pipe, "named_steps") else pipe
        if hasattr(clf, "get_feature_importance"):
            imp = np.asarray(clf.get_feature_importance(), float)
        elif hasattr(clf, "feature_importances_"):
            imp = np.asarray(clf.feature_importances_, float)
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            continue
        out[name] = imp / imp.sum() if imp.sum() else imp
    return pd.DataFrame(out, index=bundle.features)


def plot_importance_bars(df: pd.DataFrame, value_col: str, err_col: str | None, title: str, path=None):
    d = df.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.barh(d["feature"], d[value_col], color=PALETTE[0], height=0.6,
            xerr=d[err_col] if err_col else None, error_kw={"lw": 1, "capsize": 2})
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)
