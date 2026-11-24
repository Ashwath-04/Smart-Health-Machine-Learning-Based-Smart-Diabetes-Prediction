"""Dataset validation: schema, dtypes, duplicates, missing and physiologically invalid values."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


# Hard plausibility limits for a single adult measurement. Values outside these ranges are
# treated as data errors (training) or rejected (inference). Zero is handled separately for
# the columns in ``zero_as_missing`` because it encodes "not measured" in this dataset.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (1, 600),  # mg/dL, 2-hour OGTT plasma glucose
    "BloodPressure": (1, 200),  # mm Hg, diastolic
    "SkinThickness": (1, 100),  # mm, triceps skinfold
    "Insulin": (1, 1000),  # mu U/ml, 2-hour serum insulin
    "BMI": (10, 80),  # kg/m^2
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (18, 120),  # years
}


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_columns: list[str]
    unexpected_columns: list[str]
    dtypes: dict[str, str]
    non_numeric_columns: list[str]
    n_duplicate_rows: int
    explicit_missing: dict[str, int]
    zero_counts: dict[str, int]
    zero_fraction: dict[str, float]
    out_of_range: dict[str, int]
    target_distribution: dict[str, int]
    target_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_valid"] = self.is_valid
        return d


def validate_dataframe(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    zero_as_missing: list[str],
) -> ValidationReport:
    """Run all dataset checks and return a structured report (never mutates ``df``)."""
    errors: list[str] = []
    warnings: list[str] = []
    expected = list(features) + [target]

    missing_cols = [c for c in expected if c not in df.columns]
    unexpected = [c for c in df.columns if c not in expected]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    if unexpected:
        warnings.append(f"Unexpected columns will be ignored: {unexpected}")

    present = [c for c in expected if c in df.columns]
    non_numeric = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")

    explicit_missing = {c: int(df[c].isna().sum()) for c in present}
    if any(explicit_missing.values()):
        warnings.append("Explicit NaN values present; they will be imputed.")

    numeric_feats = [c for c in features if c in present and c not in non_numeric]
    zero_counts = {c: int((df[c] == 0).sum()) for c in zero_as_missing if c in numeric_feats}
    zero_fraction = {c: round(v / max(len(df), 1), 4) for c, v in zero_counts.items()}

    out_of_range: dict[str, int] = {}
    for col in numeric_feats:
        lo, hi = PLAUSIBLE_RANGES.get(col, (-np.inf, np.inf))
        values = df[col].dropna()
        if col in zero_as_missing:
            values = values[values != 0]
        n_bad = int(((values < lo) | (values > hi)).sum())
        out_of_range[col] = n_bad
        if n_bad:
            warnings.append(f"{col}: {n_bad} value(s) outside plausible range [{lo}, {hi}]")
    neg = {c: int((df[c] < 0).sum()) for c in numeric_feats}
    if any(neg.values()):
        errors.append(f"Negative values found: { {k: v for k, v in neg.items() if v} }")

    target_valid = target in df.columns and set(df[target].dropna().unique()) <= {0, 1}
    if target in df.columns and not target_valid:
        errors.append(f"Target '{target}' must be binary 0/1")
    target_dist = (
        {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()}
        if target in df.columns
        else {}
    )

    n_dup = int(df.duplicated().sum())
    if n_dup:
        warnings.append(f"{n_dup} duplicate rows")

    return ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_columns=missing_cols,
        unexpected_columns=unexpected,
        dtypes={c: str(df[c].dtype) for c in df.columns},
        non_numeric_columns=non_numeric,
        n_duplicate_rows=n_dup,
        explicit_missing=explicit_missing,
        zero_counts=zero_counts,
        zero_fraction=zero_fraction,
        out_of_range=out_of_range,
        target_distribution=target_dist,
        target_valid=target_valid,
        errors=errors,
        warnings=warnings,
    )
