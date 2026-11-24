"""Model construction, artifact persistence, inference and explanation tests."""

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.inference.nl_parser import complete_with_medians, parse_text
from src.inference.schemas import PatientFeatures
from src.models.bundle import ModelBundle
from src.models.registry import CandidateSpec
from src.models.stacking_model import BASE_LEARNERS, make_stacking


def test_stacking_structure(cfg):
    st = make_stacking(cfg, smote=True)
    assert isinstance(st, StackingClassifier)
    assert [n for n, _ in st.estimators] == list(BASE_LEARNERS)
    assert isinstance(st.final_estimator, LogisticRegression)
    assert isinstance(st.cv, StratifiedKFold) and st.stack_method == "predict_proba" and not st.passthrough
    for _, pipe in st.estimators:
        assert [s for s, _ in pipe.steps] == ["zero_to_nan", "imputer", "smote", "clf"]


def test_candidate_names(cfg):
    assert CandidateSpec("xgboost", smote=True, tuned=False).name == "xgboost_smote_default"
    assert CandidateSpec("stacking", smote=False, tuned=True).name == "stacking_nosmote_tuned"
    assert "smote" not in [s for s, _ in CandidateSpec("catboost", smote=False, tuned=False).build(cfg).steps]


def test_bundle_roundtrip_and_probabilities(tiny_bundle, split):
    _, X_te, _, _ = split
    b = ModelBundle.load(tiny_bundle)
    p = b.predict_proba(X_te)
    assert p.shape == (len(X_te),) and np.all((p >= 0) & (p <= 1))
    assert set(b.predict(X_te)) <= {0, 1}
    assert b.threshold("high_sensitivity") == 0.3
    with pytest.raises(KeyError):
        b.threshold("nonexistent")


def test_bundle_rejects_missing_columns(tiny_bundle, split):
    b = ModelBundle.load(tiny_bundle)
    with pytest.raises(ValueError, match="Missing feature"):
        b.predict_proba(split[1].drop(columns=["Age"]))


def test_bundle_load_wrong_type(tmp_path):
    import joblib

    joblib.dump({"not": "a bundle"}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        ModelBundle.load(tmp_path / "x.joblib")


def test_bundle_handles_nan_inputs(tiny_bundle):
    b = ModelBundle.load(tiny_bundle)
    row = pd.DataFrame([{**{f: 1.0 for f in b.features}, "Glucose": 120, "BMI": 30, "Age": 40,
                         "Insulin": np.nan, "SkinThickness": np.nan}])
    assert 0 <= b.predict_proba(row)[0] <= 1


def test_predictor_output_and_shap_additivity(predictor, valid_payload):
    res = predictor.predict(PatientFeatures(**valid_payload), mode="balanced")
    assert res["prediction"] in (0, 1) and 0 <= res["probability"] <= 1
    assert res["prediction"] == int(res["probability"] >= res["threshold"])
    exp = res["explanation"]
    assert len(exp["contributions"]) == 8
    total = exp["base_value"] + sum(c["shap_value"] for c in exp["contributions"])
    assert total == pytest.approx(res["probability"], abs=1e-3)  # SHAP values are real, not hard-coded
    assert "not causes" in exp["summary"]
    assert "You have diabetes" not in res["risk_statement"]


def test_predictor_modes_change_threshold(predictor, valid_payload):
    f = PatientFeatures(**valid_payload)
    t = {m: predictor.predict(f, mode=m, explain=False)["threshold"] for m in ("high_sensitivity", "balanced", "high_precision")}
    assert t["high_sensitivity"] < t["balanced"] < t["high_precision"]


def test_predictor_imputes_optional_fields(predictor, valid_payload):
    payload = {**valid_payload, "insulin": None}
    payload.pop("skin_thickness")
    res = predictor.predict(PatientFeatures(**payload), explain=True)
    assert set(res["imputed_features"]) == {"insulin", "skin_thickness"}
    imputed = {c["feature"]: c["imputed"] for c in res["explanation"]["contributions"]}
    assert imputed["Insulin"] and imputed["SkinThickness"] and not imputed["Glucose"]


def test_predictor_warns_outside_training_range(predictor, valid_payload):
    res = predictor.predict(PatientFeatures(**{**valid_payload, "age": 110}), explain=False)
    assert any("outside the training data range" in w for w in res["warnings"])


@pytest.mark.parametrize("bad", [
    {"glucose": 0}, {"glucose": -5}, {"bmi": 5}, {"age": 10}, {"age": 200}, {"pregnancies": -1},
    {"blood_pressure": 0}, {"insulin": 0}, {"diabetes_pedigree": 5}, {"glucose": float("nan")},
])
def test_schema_rejects_impossible_values(valid_payload, bad):
    with pytest.raises(ValidationError):
        PatientFeatures(**{**valid_payload, **bad})


def test_schema_rejects_extra_and_missing_fields(valid_payload):
    with pytest.raises(ValidationError):
        PatientFeatures(**{**valid_payload, "cholesterol": 200})
    p = dict(valid_payload)
    p.pop("glucose")
    with pytest.raises(ValidationError):
        PatientFeatures(**p)


def test_schema_row_mapping(valid_payload):
    row = PatientFeatures(**{**valid_payload, "insulin": None}).to_row()
    assert row["DiabetesPedigreeFunction"] == 0.45 and math.isnan(row["Insulin"])


@pytest.mark.parametrize("text,expected", [
    ("Predict diabetes risk for a 45-year-old with glucose 160 and BMI 31", {"age": 45, "glucose": 160, "bmi": 31}),
    ("age 52, blood pressure 130/85, insulin 120, 3 pregnancies, pedigree 0.6",
     {"age": 52, "blood_pressure": 85, "insulin": 120, "pregnancies": 3, "diabetes_pedigree": 0.6}),
    ("woman aged 30 never pregnant, bmi of 22.5, glucose is 99", {"age": 30, "pregnancies": 0, "bmi": 22.5, "glucose": 99}),
])
def test_nl_parser(text, expected):
    values, _ = parse_text(text)
    for k, v in expected.items():
        assert values[k] == pytest.approx(v)


def test_nl_parser_never_invents_values():
    values, _ = parse_text("please predict my risk")
    assert all(v is None for v in values.values())
    med = {k: 1.0 for k in ["pregnancies", "glucose", "blood_pressure", "bmi", "diabetes_pedigree", "age"]}
    _, filled, missing = complete_with_medians(values, med, fill=False)
    assert filled == [] and len(missing) == 6
