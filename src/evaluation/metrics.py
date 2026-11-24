"""Classification metrics, threshold analysis and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "specificity",
    "sensitivity",
    "brier",
    "ece",
    "log_loss",
    "mcc",
    "balanced_accuracy",
]


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Equal-width-bin ECE: weighted mean |observed rate - mean predicted| per bin."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    """Full metric set at a given threshold (probability metrics are threshold-free)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if both_classes else np.nan,
        "pr_auc": average_precision_score(y_true, proba) if both_classes else np.nan,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "brier": brier_score_loss(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1]),
        "mcc": matthews_corrcoef(y_true, y_pred) if both_classes else 0.0,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def threshold_table(y_true, proba, start=0.05, stop=0.95, step=0.01) -> pd.DataFrame:
    """Confusion counts and precision/recall/F1 across a grid of thresholds."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in np.round(np.arange(start, stop + 1e-9, step), 4):
        pred = proba >= t
        tp = int(np.sum(pred & (y_true == 1)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            dict(threshold=float(t), tp=tp, tn=tn, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1, specificity=spec)
        )
    return pd.DataFrame(rows)


def select_thresholds(table: pd.DataFrame, min_recall: float, min_precision: float) -> dict[str, dict]:
    """Pick the three operating points from a threshold table built on *validation* data.

    * balanced: maximum F1 (ties broken by the threshold closest to 0.5)
    * high_sensitivity: the highest threshold whose recall >= ``min_recall``
      (i.e. the best precision that still meets the sensitivity target)
    * high_precision: the lowest threshold whose precision >= ``min_precision``
      (i.e. the best recall that still meets the precision target)
    """
    out: dict[str, dict] = {}
    best_f1 = table["f1"].max()
    ties = table[np.isclose(table["f1"], best_f1)]
    bal = ties.iloc[(ties["threshold"] - 0.5).abs().argsort().iloc[0]]
    out["balanced"] = {"threshold": float(bal["threshold"]), "target_met": True, "rule": "max F1"}

    ok = table[table["recall"] >= min_recall]
    if len(ok):
        row = ok.loc[ok["threshold"].idxmax()]
        out["high_sensitivity"] = {"threshold": float(row["threshold"]), "target_met": True,
                                   "rule": f"highest threshold with recall >= {min_recall}"}
    else:
        row = table.loc[table["threshold"].idxmin()]
        out["high_sensitivity"] = {"threshold": float(row["threshold"]), "target_met": False,
                                   "rule": f"recall target {min_recall} unreachable; lowest grid threshold"}

    ok = table[(table["precision"] >= min_precision) & (table["tp"] > 0)]
    if len(ok):
        row = ok.loc[ok["threshold"].idxmin()]
        out["high_precision"] = {"threshold": float(row["threshold"]), "target_met": True,
                                 "rule": f"lowest threshold with precision >= {min_precision}"}
    else:
        row = table.loc[table["precision"].idxmax()]
        out["high_precision"] = {"threshold": float(row["threshold"]), "target_met": False,
                                 "rule": f"precision target {min_precision} unreachable; max-precision threshold"}

    # Keep the modes ordered: high sensitivity never uses a higher threshold than balanced (it
    # would be *less* sensitive), and high precision never a lower one.
    t_bal = out["balanced"]["threshold"]
    if out["high_sensitivity"]["threshold"] > t_bal:
        out["high_sensitivity"].update(threshold=t_bal, rule=out["high_sensitivity"]["rule"] + " (capped at balanced)")
    if out["high_precision"]["threshold"] < t_bal:
        out["high_precision"].update(threshold=t_bal, rule=out["high_precision"]["rule"] + " (floored at balanced)")
    return out


def bootstrap_ci(
    y_true,
    proba,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap (point, lower, upper) for ``metric_fn(y, proba)``."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], proba[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y_true, proba)), float(lo), float(hi)


def paired_bootstrap_diff(
    y_true,
    proba_a,
    proba_b,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap the difference metric(A) - metric(B) on the *same* resampled patients."""
    y_true, pa, pb = np.asarray(y_true), np.asarray(proba_a), np.asarray(proba_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(metric_fn(y_true[idx], pa[idx]) - metric_fn(y_true[idx], pb[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(metric_fn(y_true, pa) - metric_fn(y_true, pb)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float(np.mean(diffs > 0)),
    }


def thresholded(metric: str, threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    """Wrap a label metric so it can be bootstrapped on probabilities."""
    fns = {
        "f1": lambda y, p: f1_score(y, (p >= threshold).astype(int), zero_division=0),
        "recall": lambda y, p: recall_score(y, (p >= threshold).astype(int), zero_division=0),
        "precision": lambda y, p: precision_score(y, (p >= threshold).astype(int), zero_division=0),
    }
    return fns[metric]


def summarize_folds(fold_df: pd.DataFrame, group_col: str = "candidate") -> pd.DataFrame:
    """Mean and std across CV folds for every metric."""
    metrics = [c for c in METRIC_COLUMNS if c in fold_df.columns]
    agg = fold_df.groupby(group_col)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()
