"""Shared fixtures. A tiny stacking bundle is trained once per session (fast settings) so the
model, predictor and API tests do not depend on the full training run."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WARM_SHAP", "0")

from src.config import load_config  # noqa: E402
from src.data.loader import load_dataset, split_data  # noqa: E402
from src.data.preprocessing import training_medians  # noqa: E402
from src.models.bundle import ModelBundle  # noqa: E402
from src.models.calibration import ProbabilityCalibrator  # noqa: E402
from src.models.stacking_model import make_stacking  # noqa: E402
from src.training.train import quick_overrides  # noqa: E402

FAST = {"xgboost": {"n_estimators": 30}, "extratrees": {"n_estimators": 30}, "catboost": {"iterations": 30}}


@pytest.fixture(scope="session")
def cfg():
    return load_config().copy_with(quick_overrides())


@pytest.fixture(scope="session")
def raw_df(cfg):
    return load_dataset(cfg)


@pytest.fixture(scope="session")
def split(raw_df, cfg):
    return split_data(raw_df, cfg)


@pytest.fixture(scope="session")
def tiny_bundle(split, cfg, tmp_path_factory):
    X_train, _, y_train, _ = split
    model = make_stacking(cfg, base_params=FAST, smote=True).fit(X_train, y_train)
    zero = cfg["data"]["zero_as_missing"]
    medians = training_medians(X_train, zero)
    cleaned = X_train.replace({c: {0: np.nan} for c in zero})
    bundle = ModelBundle(
        name="test_stacking", model=model, calibrator=ProbabilityCalibrator("none"),
        thresholds={"balanced": 0.5, "high_sensitivity": 0.3, "high_precision": 0.7}, default_mode="balanced",
        features=cfg.features, imputation_medians=medians,
        training_ranges={c: (float(cleaned[c].min()), float(cleaned[c].max())) for c in cfg.features},
        metadata={"model_name": "test_stacking", "model_version": "test-0", "preprocessing": {"zero_as_missing": zero},
                  "thresholds": {"balanced": 0.5, "high_sensitivity": 0.3, "high_precision": 0.7}},
    )
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    bundle.save(path)
    return path


@pytest.fixture(scope="session")
def predictor(tiny_bundle, split, cfg):
    from src.explainability.shap_analysis import impute_like_training
    from src.inference.predictor import DiabetesPredictor

    X_train = split[0]
    bundle = ModelBundle.load(tiny_bundle)
    bg = impute_like_training(X_train.sample(12, random_state=0), bundle, cfg["data"]["zero_as_missing"])
    bg_path = tiny_bundle.parent / "bg.csv"
    bg.to_csv(bg_path, index=False)
    return DiabetesPredictor(tiny_bundle, background_path=bg_path, zero_as_missing=cfg["data"]["zero_as_missing"])


@pytest.fixture
def valid_payload():
    return {"pregnancies": 2, "glucose": 140, "blood_pressure": 80, "skin_thickness": 25, "insulin": 100,
            "bmi": 30.5, "diabetes_pedigree": 0.45, "age": 40}


@pytest.fixture
def small_frame():
    return pd.DataFrame({
        "Pregnancies": [0, 1, 2, 3], "Glucose": [0, 100, 120, 140], "BloodPressure": [70, 0, 80, 90],
        "SkinThickness": [0, 20, 30, 40], "Insulin": [0, 0, 100, 200], "BMI": [25.0, 0.0, 30.0, 35.0],
        "DiabetesPedigreeFunction": [0.2, 0.3, 0.4, 0.5], "Age": [25, 35, 45, 55],
    })
