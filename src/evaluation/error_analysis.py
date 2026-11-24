"""Error analysis: which patients does the model get wrong, and how do they differ?

All comparisons are descriptive. A feature pattern among errors means the model's decision
surface struggles there; it says nothing about what causes diabetes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import ZeroToNaN
from src.evaluation.plots import INK, MUTED, PALETTE, SURFACE, _save, plt

ERROR_ORDER = ["TP", "FN", "FP", "TN"]
ERROR_COLORS = {"TP": PALETTE[1], "FN": PALETTE[7], "FP": PALETTE[3], "TN": PALETTE[0]}


def error_table(X: pd.DataFrame, y: pd.Series | np.ndarray, proba: np.ndarray, threshold: float,
                zero_as_missing: list[str]) -> pd.DataFrame:
    """Per-patient table: raw features (impossible zeros as NaN), probability and error type."""
    df = ZeroToNaN(zero_as_missing).fit(X).transform(X).copy()
    y = np.asarray(y).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    df["y_true"], df["proba"], df["y_pred"] = y, proba, pred
    df["error_type"] = np.select(
        [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1)], ["TP", "FN", "FP"], "TN"
    )
    df["n_missing_inputs"] = df[zero_as_missing].isna().sum(axis=1)
    df["distance_to_threshold"] = (df["proba"] - threshold).abs()
    return df


def error_profile(err: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Median feature values (and counts) for each error type."""
    prof = err.groupby("error_type")[features + ["proba", "n_missing_inputs"]].median()
    prof.insert(0, "n", err["error_type"].value_counts())
    return prof.reindex([e for e in ERROR_ORDER if e in prof.index])


def borderline_cases(err: pd.DataFrame, width: float = 0.10) -> pd.DataFrame:
    """Patients whose probability lies within ``width`` of the threshold."""
    return err[err["distance_to_threshold"] <= width].sort_values("distance_to_threshold")


def missed_vs_caught(err: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compare false negatives with true positives (both truly diabetic)."""
    fn, tp = err[err.error_type == "FN"], err[err.error_type == "TP"]
    rows = []
    for f in features:
        rows.append({"feature": f, "median_FN": fn[f].median(), "median_TP": tp[f].median(),
                     "median_TN": err.loc[err.error_type == "TN", f].median(),
                     "FN_minus_TP": fn[f].median() - tp[f].median()})
    return pd.DataFrame(rows)


def summarize_errors(err: pd.DataFrame, threshold: float) -> dict:
    counts = err["error_type"].value_counts().to_dict()
    fn, tp = err[err.error_type == "FN"], err[err.error_type == "TP"]
    fp, tn = err[err.error_type == "FP"], err[err.error_type == "TN"]
    return {
        "threshold": threshold,
        "counts": {k: int(counts.get(k, 0)) for k in ERROR_ORDER},
        "fn_share_with_glucose_below_126": float((fn["Glucose"] < 126).mean()) if len(fn) else None,
        "tp_share_with_glucose_below_126": float((tp["Glucose"] < 126).mean()) if len(tp) else None,
        "fp_share_with_glucose_at_least_126": float((fp["Glucose"] >= 126).mean()) if len(fp) else None,
        "tn_share_with_glucose_at_least_126": float((tn["Glucose"] >= 126).mean()) if len(tn) else None,
        "fn_median_age": float(fn["Age"].median()) if len(fn) else None,
        "tp_median_age": float(tp["Age"].median()) if len(tp) else None,
        "fn_median_bmi": float(fn["BMI"].median()) if len(fn) else None,
        "tp_median_bmi": float(tp["BMI"].median()) if len(tp) else None,
        "mean_missing_inputs_by_type": err.groupby("error_type")["n_missing_inputs"].mean().round(3).to_dict(),
        "n_borderline_0.10": int((err["distance_to_threshold"] <= 0.10).sum()),
        "error_rate_borderline_0.10": float(
            err.loc[err["distance_to_threshold"] <= 0.10, "error_type"].isin(["FP", "FN"]).mean()
        ) if (err["distance_to_threshold"] <= 0.10).any() else None,
        "error_rate_confident_gt_0.30": float(
            err.loc[err["distance_to_threshold"] > 0.30, "error_type"].isin(["FP", "FN"]).mean()
        ) if (err["distance_to_threshold"] > 0.30).any() else None,
    }


def plot_error_features(err: pd.DataFrame, features: list[str], path=None, title="Feature values by outcome type"):
    """Strip plots of each feature split by TP/FN/FP/TN."""
    n = len(features)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.3 * nrows), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, f in zip(axes.ravel(), features):
        for i, et in enumerate(ERROR_ORDER):
            vals = err.loc[err.error_type == et, f].dropna()
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=10, alpha=0.6,
                       color=ERROR_COLORS[et], linewidths=0)
            if len(vals):
                ax.hlines(vals.median(), i - 0.3, i + 0.3, color=INK, lw=1.6)
        ax.set_xticks(range(4), [f"{e}\n(n={int((err.error_type == e).sum())})" for e in ERROR_ORDER], fontsize=8)
        ax.set_title(f, fontsize=10)
        ax.grid(axis="x", visible=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}  (black bar = median; FN = diabetic predicted negative)", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_error_probability(err: pd.DataFrame, threshold: float, path=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.linspace(0, 1, 26)
    bottom = np.zeros(len(bins) - 1)
    for et in ERROR_ORDER:
        counts, _ = np.histogram(err.loc[err.error_type == et, "proba"], bins=bins)
        ax.bar(bins[:-1], counts, width=np.diff(bins), bottom=bottom, align="edge", color=ERROR_COLORS[et],
               edgecolor=SURFACE, linewidth=1, label=et)
        bottom += counts
    ax.axvline(threshold, color=INK, lw=1.2, ls="--")
    ax.text(threshold, 1.01, f"threshold {threshold:.2f}", ha="center", fontsize=8, color=MUTED,
            transform=ax.get_xaxis_transform())
    ax.set(xlabel="Predicted probability", ylabel="Patients", title="Where errors occur on the probability scale")
    ax.legend(ncol=4, loc="upper right")
    return _save(fig, path)
