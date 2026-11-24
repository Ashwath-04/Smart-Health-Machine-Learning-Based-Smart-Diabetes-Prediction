"""Preprocessing, SMOTE placement, metrics and calibration tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from src.data.preprocessing import ZeroToNaN, build_preprocessor, training_medians
from src.evaluation.metrics import compute_metrics, select_thresholds, threshold_table
from src.models.calibration import ProbabilityCalibrator, select_calibration
from src.models.common import build_pipeline

ZERO = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def test_zero_to_nan_only_in_listed_columns(small_frame):
    out = ZeroToNaN(ZERO).fit(small_frame).transform(small_frame)
    assert out.loc[0, "Pregnancies"] == 0  # genuine zero preserved
    assert np.isnan(out.loc[0, "Glucose"]) and np.isnan(out.loc[1, "BMI"]) and np.isnan(out.loc[1, "Insulin"])
    assert out.loc[1, "Glucose"] == 100
    assert small_frame.loc[0, "Glucose"] == 0  # input not mutated


def test_zero_to_nan_unknown_column(small_frame):
    with pytest.raises(ValueError):
        ZeroToNaN(["NotAColumn"]).fit(small_frame)


def test_imputer_uses_training_statistics_only(small_frame):
    pp = build_preprocessor(ZERO).fit(small_frame)
    other = small_frame.copy()
    other["Glucose"] = [0, 0, 0, 0]  # all missing in "test" data
    out = pp.transform(other)
    assert not out.isna().any().any()
    # median of non-zero training glucose (100, 120, 140) = 120, not anything from `other`
    assert (out["Glucose"] == 120).all()


def test_training_medians_ignore_zeros(small_frame):
    med = training_medians(small_frame, ZERO)
    assert med["Insulin"] == 150.0  # (100, 200) -> zeros excluded
    assert med["Pregnancies"] == 1.5  # zeros are valid here


def test_scaling_only_when_requested(small_frame):
    unscaled = build_preprocessor(ZERO, scale=False).fit_transform(small_frame)
    scaled = build_preprocessor(ZERO, scale=True).fit_transform(small_frame)
    assert unscaled["Age"].tolist() == [25, 35, 45, 55]
    assert abs(scaled["Age"].mean()) < 1e-9


def test_preprocessor_output_is_dataframe_with_names(small_frame):
    out = build_preprocessor(ZERO).fit_transform(small_frame)
    assert list(out.columns) == list(small_frame.columns)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Records how many rows it was trained on and asked to predict."""

    def fit(self, X, y):
        self.n_fit_, self.fit_pos_rate_ = len(X), float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        self.n_pred_ = len(X)
        return np.tile([0.5, 0.5], (len(X), 1))


def test_smote_resamples_training_data_only(split, cfg):
    X_tr, X_te, y_tr, _ = split
    pipe = build_pipeline(RecordingClassifier(), cfg, smote=True).fit(X_tr, y_tr)
    clf = pipe.named_steps["clf"]
    assert clf.n_fit_ == 2 * int((y_tr == 0).sum())  # minority oversampled to majority size
    assert clf.fit_pos_rate_ == pytest.approx(0.5)
    pipe.predict_proba(X_te)
    assert clf.n_pred_ == len(X_te)  # no synthetic rows at prediction time


def test_no_smote_pipeline_keeps_class_balance(split, cfg):
    X_tr, _, y_tr, _ = split
    clf = build_pipeline(RecordingClassifier(), cfg, smote=False).fit(X_tr, y_tr).named_steps["clf"]
    assert clf.n_fit_ == len(X_tr)
    assert clf.fit_pos_rate_ == pytest.approx(y_tr.mean())


def test_compute_metrics_known_values():
    y = np.array([1, 1, 0, 0, 1, 0])
    p = np.array([0.9, 0.4, 0.6, 0.1, 0.8, 0.2])
    m = compute_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 2)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)


def test_threshold_selection_rules():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    table = threshold_table(y, p)
    sel = select_thresholds(table, min_recall=0.9, min_precision=0.8)
    hs = table.set_index("threshold").loc[sel["high_sensitivity"]["threshold"]]
    hp = table.set_index("threshold").loc[sel["high_precision"]["threshold"]]
    assert hs["recall"] >= 0.9 and hp["precision"] >= 0.8
    assert sel["high_sensitivity"]["threshold"] <= sel["balanced"]["threshold"] <= sel["high_precision"]["threshold"]
    assert table.set_index("threshold").loc[sel["balanced"]["threshold"], "f1"] == pytest.approx(table["f1"].max())


def test_calibration_improves_miscalibrated_scores():
    rng = np.random.default_rng(1)
    true_p = rng.uniform(0, 1, 3000)
    y = (rng.uniform(0, 1, 3000) < true_p).astype(int)
    distorted = true_p ** 3  # systematically under-confident for positives
    groups = np.arange(3000)
    method, scores = select_calibration(distorted, y, groups, ["none", "sigmoid", "isotonic"], 0.002)
    assert method in {"sigmoid", "isotonic"}
    assert scores[method] < scores["none"]
    cal = ProbabilityCalibrator(method).fit(distorted, y)
    out = cal.transform(distorted)
    assert out.min() >= 0 and out.max() <= 1


def test_calibrator_none_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(ProbabilityCalibrator("none").fit(p, [0, 1, 1]).transform(p), p)
    with pytest.raises(ValueError):
        ProbabilityCalibrator("bogus")
