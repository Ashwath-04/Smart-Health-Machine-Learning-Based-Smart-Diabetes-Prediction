"""Post-hoc probability calibration fitted on out-of-fold training predictions.

The calibrator maps a model's raw ``P(y=1)`` to a calibrated probability. It is fitted on
out-of-fold (OOF) predictions from the training set, which is equivalent to sklearn's
``CalibratedClassifierCV(ensemble=False)`` but reuses the OOF predictions the CV stage
already produced. Whether calibration is adopted is decided by a *nested* CV estimate of
the Brier score, never by the test set.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator:
    """``method`` is one of ``none``, ``sigmoid`` (Platt scaling on the logit) or ``isotonic``."""

    def __init__(self, method: str = "none"):
        if method not in {"none", "sigmoid", "isotonic"}:
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self._model = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        proba, y = np.asarray(proba, float), np.asarray(y, int)
        if self.method == "sigmoid":
            self._model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(proba).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(proba, y)
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, float)
        if self.method == "none":
            return proba
        if self._model is None:
            raise RuntimeError("Calibrator is not fitted")
        if self.method == "sigmoid":
            return self._model.predict_proba(_logit(proba).reshape(-1, 1))[:, 1]
        return self._model.predict(proba)

    def __repr__(self) -> str:
        return f"ProbabilityCalibrator(method={self.method!r})"


def nested_calibrated_oof(
    proba: np.ndarray, y: np.ndarray, groups: np.ndarray, method: str, n_splits: int = 5
) -> np.ndarray:
    """Calibrated OOF probabilities where the calibrator never saw the evaluated patient.

    ``groups`` holds the patient id, so stacked OOF predictions from repeated CV of the same
    patient never end up on both sides of a calibration split.
    """
    out = np.empty_like(np.asarray(proba, float))
    for tr, te in GroupKFold(n_splits=n_splits).split(proba, y, groups):
        cal = ProbabilityCalibrator(method).fit(proba[tr], y[tr])
        out[te] = cal.transform(proba[te])
    return out


def select_calibration(
    proba: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    methods: list[str],
    min_improvement: float,
) -> tuple[str, dict[str, float]]:
    """Return (chosen method, nested-CV Brier per method). Prefers ``none`` unless a method
    improves the Brier score by at least ``min_improvement``."""
    scores = {}
    for m in methods:
        p = proba if m == "none" else nested_calibrated_oof(proba, y, groups, m)
        scores[m] = float(brier_score_loss(y, p))
    baseline = scores.get("none", min(scores.values()))
    best = min(scores, key=scores.get)
    chosen = best if baseline - scores[best] >= min_improvement else "none"
    return chosen, scores
