"""Inference service: validated input -> calibrated probability -> thresholded class -> SHAP.

Training and inference are separated: this module only loads artifacts written by
``scripts/train.py`` and never refits anything.
"""

from __future__ import annotations

import json
import logging
import math
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Config, load_config
from src.inference.explanation import DISCLAIMER, SHAP_NOTE, risk_statement, summarize
from src.inference.nl_parser import complete_with_medians, parse_text
from src.inference.schemas import COLUMN_TO_FIELD, DISPLAY, FIELD_TO_COLUMN, PatientFeatures
from src.models.bundle import ModelBundle

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


class DiabetesPredictor:
    def __init__(self, model_path: str | Path, metadata_path: str | Path | None = None,
                 background_path: str | Path | None = None, train_csv: str | Path | None = None,
                 zero_as_missing: list[str] | None = None, background_size: int = 50, seed: int = 42):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotLoadedError(f"Model artifact not found at {model_path}. Run `python scripts/train.py`.")
        self.bundle = ModelBundle.load(model_path)
        self.metadata: dict[str, Any] = dict(self.bundle.metadata)
        if metadata_path and Path(metadata_path).exists():
            # JSON metadata may contain test metrics added after training
            self.metadata.update(json.loads(Path(metadata_path).read_text(encoding="utf-8")))
        self.zero_as_missing = zero_as_missing or self.metadata.get("preprocessing", {}).get("zero_as_missing", [])
        self._background_path = Path(background_path) if background_path else None
        self._train_csv = Path(train_csv) if train_csv else None
        self._background_size, self._seed = background_size, seed

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "DiabetesPredictor":
        cfg = cfg or load_config()
        models_dir = cfg.path("paths", "models_dir")
        return cls(
            models_dir / "model.joblib",
            metadata_path=models_dir / "model_metadata.json",
            background_path=models_dir / "shap_background.csv",
            train_csv=cfg.path("data", "processed_dir") / "train.csv",
            zero_as_missing=cfg["data"]["zero_as_missing"],
            background_size=cfg["explainability"]["background_size"],
            seed=cfg.seed,
        )

    # ------------------------------------------------------------------ info
    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", self.bundle.name))

    def info(self) -> dict[str, Any]:
        keys = ["model_name", "display_name", "model_version", "dataset_version", "training_date", "features",
                "random_seed", "preprocessing", "smote", "calibration", "thresholds", "threshold_rules",
                "default_threshold_mode", "cv_metrics", "test_metrics", "selection", "meta_learner",
                "hyperparameters", "library_versions"]
        return {k: self.metadata[k] for k in keys if k in self.metadata}

    # ------------------------------------------------------------------ SHAP
    @cached_property
    def explainer(self):
        from src.explainability.shap_analysis import ShapExplainer, background_sample, impute_like_training

        if self._background_path and self._background_path.exists():
            bg = pd.read_csv(self._background_path)
        elif self._train_csv and self._train_csv.exists():
            train = pd.read_csv(self._train_csv, index_col="row_id")
            bg = background_sample(impute_like_training(train, self.bundle, self.zero_as_missing),
                                   train[self.metadata.get("target", "Outcome")], self._background_size, self._seed)
        else:  # last resort: a single median patient (explanations still additive, less informative)
            bg = pd.DataFrame([self.bundle.imputation_medians])[self.bundle.features]
        return ShapExplainer(self.bundle, bg)

    def warmup(self) -> None:
        """Run one explained prediction so the first real request is not slow (SHAP JIT)."""
        med = self.medians_by_field()
        self.predict(PatientFeatures(**{k: (int(round(v)) if k in {"pregnancies", "age"} else v)
                                        for k, v in med.items()}), explain=True)

    # ------------------------------------------------------------------ predict
    def _warnings(self, row: dict[str, float]) -> list[str]:
        warns = []
        for col, (lo, hi) in self.bundle.training_ranges.items():
            v = row.get(col)
            if v is not None and not math.isnan(v) and (v < lo or v > hi):
                warns.append(f"{DISPLAY[col][0]} = {v:g} is outside the training data range [{lo:g}, {hi:g}]; "
                             "the model is extrapolating and the estimate is less reliable.")
        return warns

    def predict(self, features: PatientFeatures, mode: str | None = None, explain: bool = True) -> dict[str, Any]:
        mode = mode or self.bundle.default_mode
        threshold = self.bundle.threshold(mode)
        row = features.to_row()
        imputed = [c for c, v in row.items() if math.isnan(v)]
        X = pd.DataFrame([row])[self.bundle.features]
        proba = float(self.bundle.predict_proba(X)[0])
        pred = int(proba >= threshold)
        result: dict[str, Any] = {
            "prediction": pred,
            "label": "higher likelihood" if pred else "lower likelihood",
            "probability": round(proba, 4),
            "threshold": threshold,
            "threshold_mode": mode,
            "model_version": self.model_version,
            "risk_statement": risk_statement(pred),
            "imputed_features": [COLUMN_TO_FIELD[c] for c in imputed],
            "warnings": self._warnings(row),
            "explanation": None,
            "disclaimer": DISCLAIMER,
        }
        if explain:
            result["explanation"] = self.explain(X, imputed, proba, threshold, mode)
        return result

    def explain(self, X: pd.DataFrame, imputed: list[str], proba: float, threshold: float, mode: str) -> dict:
        from src.explainability.shap_analysis import contributions, impute_like_training

        X_imp = impute_like_training(X, self.bundle, self.zero_as_missing)
        exp = self.explainer.explain(X_imp)[0]
        contribs = contributions(exp, imputed=imputed)
        for c in contribs:
            c["display_name"], c["unit"] = DISPLAY[c["feature"]]
        return {
            "method": "SHAP exact (model-agnostic) on calibrated probability",
            "base_value": float(exp.base_values),
            "contributions": contribs,
            "summary": summarize(proba, threshold, mode, contribs),
            "note": SHAP_NOTE,
        }

    def predict_batch(self, df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
        """Batch scoring with canonical column names (zeros in physiological columns = missing)."""
        p = self.bundle.predict_proba(df)
        t = self.bundle.threshold(mode)
        return pd.DataFrame({"probability": p, "prediction": (p >= t).astype(int), "threshold": t}, index=df.index)

    # ------------------------------------------------------------------ natural language
    def medians_by_field(self) -> dict[str, float]:
        return {COLUMN_TO_FIELD[c]: v for c, v in self.bundle.imputation_medians.items()}

    def parse(self, text: str, fill_missing: bool = True) -> dict[str, Any]:
        values, notes = parse_text(text)
        completed, filled, missing = complete_with_medians(values, self.medians_by_field(), fill_missing)
        if filled:
            notes.append("Not mentioned, filled with the training-set median: " + ", ".join(filled) + ".")
        return {"extracted": completed, "filled_with_training_median": filled, "missing_required": missing,
                "notes": notes}

    @staticmethod
    def features_from_parsed(parsed: dict[str, Any]) -> PatientFeatures:
        vals = {k: v for k, v in parsed["extracted"].items() if k in FIELD_TO_COLUMN}
        for k in ("pregnancies", "age"):
            if vals.get(k) is not None:
                vals[k] = int(round(vals[k]))
        return PatientFeatures(**vals)
