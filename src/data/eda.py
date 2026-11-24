"""Descriptive exploratory analysis (no diagnostic or causal claims)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.preprocessing import ZeroToNaN
from src.evaluation.plots import CLASS_COLORS, CLASS_LABELS, INK_2, MUTED, SURFACE, _save, plt


def clean_view(df: pd.DataFrame, zero_as_missing: list[str]) -> pd.DataFrame:
    """Copy of ``df`` with impossible zeros shown as NaN (for honest descriptive stats)."""
    out = df.copy()
    feats = [c for c in df.columns if c != "Outcome"]
    out[feats] = ZeroToNaN(zero_as_missing).fit(df[feats]).transform(df[feats])
    return out


def describe_by_class(df: pd.DataFrame, target: str = "Outcome") -> pd.DataFrame:
    """Median/IQR per class plus a Mann-Whitney U test and effect size (rank-biserial).

    Descriptive association only: a difference in distributions says nothing about causation.
    """
    rows = []
    for col in [c for c in df.columns if c != target]:
        a = df.loc[df[target] == 0, col].dropna()
        b = df.loc[df[target] == 1, col].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "feature": col,
                "n_non_diabetic": len(a),
                "n_diabetic": len(b),
                "median_non_diabetic": a.median(),
                "median_diabetic": b.median(),
                "iqr_non_diabetic": f"{a.quantile(.25):.1f}-{a.quantile(.75):.1f}",
                "iqr_diabetic": f"{b.quantile(.25):.1f}-{b.quantile(.75):.1f}",
                "mann_whitney_p": p,
                # rank-biserial: P(diabetic > non-diabetic) - P(diabetic < non-diabetic)
                "rank_biserial": 1 - 2 * u / (len(a) * len(b)),
                "point_biserial_r": stats.pointbiserialr(df.loc[df[col].notna(), target], df[col].dropna())[0],
            }
        )
    return pd.DataFrame(rows).sort_values("rank_biserial", key=np.abs, ascending=False)


def outlier_summary(df: pd.DataFrame, target: str = "Outcome") -> pd.DataFrame:
    """Tukey-fence (1.5 IQR) outlier counts on non-missing values."""
    rows = []
    for col in [c for c in df.columns if c != target]:
        s = df[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        rows.append({"feature": col, "lower_fence": lo, "upper_fence": hi,
                     "n_low": int((s < lo).sum()), "n_high": int((s > hi).sum()),
                     "min": s.min(), "max": s.max()})
    return pd.DataFrame(rows)


def plot_target_distribution(df, target="Outcome", path=None):
    counts = df[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar([CLASS_LABELS[i] for i in counts.index], counts.values, color=[CLASS_COLORS[i] for i in counts.index],
           width=0.55)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 6, f"{v}  ({v / counts.sum():.1%})", ha="center", fontsize=9, color=INK_2)
    ax.set(ylabel="Patients", title="Target distribution (Outcome)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, path)


def plot_missingness(raw: pd.DataFrame, zero_as_missing: list[str], path=None):
    frac = (raw[zero_as_missing] == 0).mean().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(frac.index, frac.values, color="#2a78d6", height=0.55)
    for i, v in enumerate(frac.values):
        ax.text(v + 0.005, i, f"{v:.1%}  ({int(round(v * len(raw)))})", va="center", fontsize=9, color=INK_2)
    ax.set(xlabel="Fraction of rows recorded as 0 (treated as missing)", title="Physiologically impossible zeros")
    ax.set_xlim(0, max(frac.max() * 1.3, 0.1))
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_class_histograms(df, target="Outcome", path=None):
    feats = [c for c in df.columns if c != target]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, col in zip(axes.ravel(), feats):
        data = df[col].dropna()
        bins = np.histogram_bin_edges(data, bins=25)
        for cls in (0, 1):
            ax.hist(df.loc[df[target] == cls, col].dropna(), bins=bins, alpha=0.6, density=True,
                    color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(col, fontsize=10)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Feature distributions by outcome (zeros-as-missing removed; densities)", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_boxplots(df, target="Outcome", path=None):
    feats = [c for c in df.columns if c != target]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, col in zip(axes.ravel(), feats):
        groups = [df.loc[df[target] == cls, col].dropna() for cls in (0, 1)]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.55, medianprops={"color": "#0b0b0b"},
                        flierprops={"marker": "o", "ms": 2.5, "mec": MUTED})
        for patch, cls in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(CLASS_COLORS[cls])
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2], ["Non-diab.", "Diabetic"], fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Box plots by outcome (Tukey whiskers; dots = outliers)", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def plot_correlation(df, path=None, method="spearman"):
    corr = df.corr(method=method)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for (i, j), v in np.ndenumerate(corr.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{method.capitalize()} correlation (pairwise complete observations)")
    return _save(fig, path)


def plot_pairs(df, features: list[str], target="Outcome", path=None):
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(2.3 * n, 2.3 * n))
    for i, fy in enumerate(features):
        for j, fx in enumerate(features):
            ax = axes[i, j]
            if i == j:
                for cls in (0, 1):
                    ax.hist(df.loc[df[target] == cls, fx].dropna(), bins=20, alpha=0.6, density=True,
                            color=CLASS_COLORS[cls])
            else:
                for cls in (0, 1):
                    d = df[df[target] == cls]
                    ax.scatter(d[fx], d[fy], s=6, alpha=0.5, color=CLASS_COLORS[cls], linewidths=0)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(fx, fontsize=8)
            if j == 0:
                ax.set_ylabel(fy, fontsize=8)
    fig.suptitle("Pairwise relationships (blue = non-diabetic, orange = diabetic)", fontsize=12)
    fig.tight_layout()
    return _save(fig, path)


def run_eda(raw: pd.DataFrame, zero_as_missing: list[str], figures_dir: Path, metrics_dir: Path) -> dict:
    """Generate all EDA figures and tables. Returns summary tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    clean = clean_view(raw, zero_as_missing)
    plot_target_distribution(raw, path=figures_dir / "eda_target_distribution.png")
    plot_missingness(raw, zero_as_missing, path=figures_dir / "eda_missing_zeros.png")
    plot_class_histograms(clean, path=figures_dir / "eda_histograms_by_class.png")
    plot_boxplots(clean, path=figures_dir / "eda_boxplots_by_class.png")
    plot_correlation(clean, path=figures_dir / "eda_correlation_spearman.png")
    plot_pairs(clean, ["Glucose", "BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction"],
               path=figures_dir / "eda_pairplot.png")
    by_class = describe_by_class(clean)
    outliers = outlier_summary(clean)
    by_class.to_csv(metrics_dir / "eda_feature_by_class.csv", index=False)
    outliers.to_csv(metrics_dir / "eda_outliers.csv", index=False)
    clean.describe().T.to_csv(metrics_dir / "eda_describe_clean.csv")
    return {"by_class": by_class, "outliers": outliers}
