"""Deployable model artifact: fitted pipeline + calibrator + thresholds + metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.calibration import ProbabilityCalibrator


@dataclass
class ModelBundle:
    """Everything needed to reproduce a prediction, serialized as one joblib file.

    ``model`` is the fitted estimator (for the deployed model: a ``StackingClassifier`` whose
    base estimators are imblearn pipelines with the fitted imputer and SMOTE step).
    """

    name: str
    model: Any
    calibrator: ProbabilityCalibrator
    thresholds: dict[str, float]
    default_mode: str
    features: list[str]
    imputation_medians: dict[str, float]
    training_ranges: dict[str, tuple[float, float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated ``P(Outcome=1)`` for each row of ``X`` (raw feature columns)."""
        X = self._prepare(X)
        raw = self.model.predict_proba(X)[:, 1]
        return self.calibrator.transform(raw)

    def predict(self, X: pd.DataFrame, mode: str | None = None) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold(mode)).astype(int)

    def threshold(self, mode: str | None = None) -> float:
        mode = mode or self.default_mode
        if mode not in self.thresholds:
            raise KeyError(f"Unknown threshold mode '{mode}'. Available: {sorted(self.thresholds)}")
        return float(self.thresholds[mode])

    def _prepare(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X, dtype=float), columns=self.features)
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        return X[self.features].astype(float)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return obj
