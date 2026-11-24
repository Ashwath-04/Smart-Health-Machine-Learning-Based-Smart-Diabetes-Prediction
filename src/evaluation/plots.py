"""Evaluation figures (matplotlib, saved as PNG).

Colours follow the entity, never its rank: each model family and each class has one fixed
colour across every figure (validated categorical palette, fixed slot order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

# Validated categorical palette (light mode), fixed slot order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

FAMILY_COLORS = {
    "stacking": PALETTE[0],
    "xgboost": PALETTE[1],
    "extratrees": PALETTE[2],
    "catboost": PALETTE[3],
    "random_forest": PALETTE[4],
    "logistic_regression": PALETTE[5],
    "soft_voting": PALETTE[6],
}
CLASS_COLORS = {0: PALETTE[0], 1: PALETTE[1]}
CLASS_LABELS = {0: "Non-diabetic (0)", 1: "Diabetic (1)"}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2,
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "figure.dpi": 110,
        }
    )


apply_style()


def family_of(name: str) -> str:
    for fam in FAMILY_COLORS:
        if name.startswith(fam):
            return fam
    return name


def color_for(name: str) -> str:
    return FAMILY_COLORS.get(family_of(name), MUTED)


def linestyle_for(name: str) -> str:
    return "--" if "_nosmote" in name else "-"


def _save(fig, path: Path | None):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curves(curves: dict[str, tuple], labels: dict[str, str] | None = None, path=None, title="ROC curves"):
    """``curves``: name -> (y_true, proba)."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score

        ax.plot(fpr, tpr, color=color_for(name), ls=linestyle_for(name),
                label=f"{(labels or {}).get(name, name)}  (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":")
    ax.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)", title=title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_pr_curves(curves: dict[str, tuple], labels: dict[str, str] | None = None, path=None, title="Precision-recall curves"):
    from sklearn.metrics import average_precision_score

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    prevalence = None
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        prevalence = float(np.mean(y))
        ax.plot(rec, prec, color=color_for(name), ls=linestyle_for(name),
                label=f"{(labels or {}).get(name, name)}  (AP {average_precision_score(y, p):.3f})")
    if prevalence is not None:
        ax.axhline(prevalence, color=AXIS, lw=1, ls=":")
        ax.text(0.01, prevalence + 0.01, f"prevalence {prevalence:.2f}", color=MUTED, fontsize=8)
    ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", title=title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_calibration_curves(curves: dict[str, tuple], labels: dict[str, str] | None = None, path=None,
                            n_bins: int = 10, title="Calibration (reliability) curves"):
    from sklearn.metrics import brier_score_loss

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":", label="Perfectly calibrated")
    for name, (y, p) in curves.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=5, color=color_for(name), ls=linestyle_for(name),
                label=f"{(labels or {}).get(name, name)}  (Brier {brier_score_loss(y, p):.3f})")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction diabetic", title=title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_confusion_matrices(items: dict[str, tuple], labels: dict[str, str] | None = None, path=None, ncols: int = 3):
    """``items``: name -> (y_true, y_pred). Single-hue sequential shading, counts annotated."""
    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax, (name, (y, yp)) in zip(axes.ravel(), items.items()):
        cm = confusion_matrix(y, yp, labels=[0, 1])
        ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#104281"]))
        ax.grid(False)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=13,
                    color="white" if v > cm.max() * 0.55 else INK)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title((labels or {}).get(name, name), fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def plot_metric_bars(df: pd.DataFrame, metric: str, name_col: str = "candidate", err_col: str | None = None,
                     labels: dict[str, str] | None = None, path=None, title: str | None = None):
    """Horizontal bars (one per model) for one metric, sorted, with optional error bars."""
    d = df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.barh(y, d[metric], color=[color_for(n) for n in d[name_col]], height=0.62,
            xerr=d[err_col] if err_col else None, error_kw={"ecolor": INK_2, "lw": 1, "capsize": 2})
    errs = d[err_col].to_numpy() if err_col else np.zeros(len(d))
    for yi, (v, n, e) in enumerate(zip(d[metric], d[name_col], errs)):
        if "_nosmote" in n:
            ax.patches[yi].set_hatch("////")
            ax.patches[yi].set_edgecolor(SURFACE)
        ax.text(v + e + 0.008, yi, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(y, [(labels or {}).get(n, n) for n in d[name_col]], fontsize=8)
    ax.set_xlim(0, min(1.05, d[metric].max() + errs.max() + 0.1))
    ax.grid(axis="y", visible=False)
    ax.set_title(title or metric)
    ax.text(1.0, -0.6 / len(d) - 0.08, "hatched = no SMOTE", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    return _save(fig, path)


def plot_cv_boxplots(fold_df: pd.DataFrame, metric: str, order: Iterable[str], labels: dict[str, str] | None = None,
                     path=None):
    order = list(order)
    data = [fold_df.loc[fold_df["candidate"] == c, metric].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.2))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                    medianprops={"color": INK, "lw": 1.5}, whiskerprops={"color": AXIS},
                    capprops={"color": AXIS}, flierprops={"marker": "o", "ms": 3, "mec": MUTED})
    for patch, c in zip(bp["boxes"], order):
        patch.set_facecolor(color_for(c))
        patch.set_alpha(0.55 if "_nosmote" in c else 0.9)
        patch.set_edgecolor(color_for(c))
    ax.set_yticks(range(1, len(order) + 1), [(labels or {}).get(c, c) for c in order], fontsize=8)
    ax.set_title(f"Cross-validation fold distribution: {metric}")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_threshold_tradeoff(table: pd.DataFrame, chosen: dict[str, float] | None = None, path=None,
                            title="Threshold trade-off (validation / OOF predictions)"):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for col, c in [("precision", PALETTE[0]), ("recall", PALETTE[1]), ("f1", PALETTE[2]), ("specificity", PALETTE[6])]:
        ax.plot(table["threshold"], table[col], color=c, label=col.capitalize() if col != "f1" else "F1")
    for mode, t in (chosen or {}).items():
        ax.axvline(t, color=MUTED, lw=1, ls="--")
        ax.text(t - 0.006, 0.03, f"{mode.replace('_', ' ')} ({t:.2f})", rotation=90, ha="right", va="bottom",
                fontsize=8, color=INK_2, transform=ax.get_xaxis_transform())
    ax.set(xlabel="Decision threshold", ylabel="Score", title=title)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, path)


def plot_probability_histogram(y_true, proba, threshold: float, path=None, title="Predicted probability by true class"):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 26)
    for cls in (0, 1):
        ax.hist(np.asarray(proba)[np.asarray(y_true) == cls], bins=bins, alpha=0.65, color=CLASS_COLORS[cls],
                label=CLASS_LABELS[cls], edgecolor=SURFACE, linewidth=1)
    ax.axvline(threshold, color=INK, lw=1.2, ls="--")
    ax.text(threshold, 1.01, f"threshold {threshold:.2f}", ha="center", fontsize=8, color=INK_2,
            transform=ax.get_xaxis_transform())
    ax.set(xlabel="Predicted probability of diabetes", ylabel="Patients", title=title)
    ax.legend()
    return _save(fig, path)
