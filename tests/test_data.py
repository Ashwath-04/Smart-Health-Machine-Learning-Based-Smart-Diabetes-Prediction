"""Data loading, schema validation and split tests."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import CANONICAL_COLUMNS, DataIntegrityError, load_raw, sha256_of, split_data
from src.data.validation import validate_dataframe


def test_load_raw_canonical_schema(raw_df):
    assert list(raw_df.columns) == CANONICAL_COLUMNS
    assert raw_df.shape == (768, 9)
    assert set(raw_df["Outcome"].unique()) == {0, 1}


def test_checksum_pinned(cfg):
    assert sha256_of(cfg.path("data", "raw_path")) == cfg["data"]["sha256"]


def test_checksum_mismatch_raises(cfg, tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("1,2,3,4,5,6,7,8,1\n")
    with pytest.raises(DataIntegrityError):
        load_raw(p, verify_sha256=cfg["data"]["sha256"])


def test_load_raw_with_header(raw_df, tmp_path):
    p = tmp_path / "kaggle.csv"
    raw_df.to_csv(p, index=False)
    df = load_raw(p)
    pd.testing.assert_frame_equal(df, raw_df)


def test_missing_file_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_raw(tmp_path / "nope.csv")


def test_validation_on_real_data(raw_df, cfg):
    rep = validate_dataframe(raw_df, cfg.features, cfg.target, cfg["data"]["zero_as_missing"])
    assert rep.is_valid
    assert rep.n_duplicate_rows == 0
    assert rep.target_distribution == {"0": 500, "1": 268}
    # documented counts of physiologically impossible zeros
    assert rep.zero_counts == {"Glucose": 5, "BloodPressure": 35, "SkinThickness": 227, "Insulin": 374, "BMI": 11}
    assert sum(rep.explicit_missing.values()) == 0


def test_validation_detects_problems(raw_df, cfg):
    bad = raw_df.head(20).copy()
    bad.loc[0, "BMI"] = -3
    bad.loc[1, "Outcome"] = 2
    bad = pd.concat([bad, bad.iloc[[5]]])
    rep = validate_dataframe(bad.drop(columns=["Age"]), cfg.features, cfg.target, cfg["data"]["zero_as_missing"])
    assert not rep.is_valid
    msgs = " ".join(rep.errors)
    assert "Missing required columns" in msgs and "Negative" in msgs and "binary" in msgs
    assert rep.n_duplicate_rows == 1


def test_validation_flags_non_numeric_and_out_of_range(raw_df, cfg):
    bad = raw_df.head(10).copy()
    bad["Glucose"] = bad["Glucose"].astype(object)
    bad.loc[0, "Glucose"] = "high"
    rep = validate_dataframe(bad, cfg.features, cfg.target, cfg["data"]["zero_as_missing"])
    assert "Glucose" in rep.non_numeric_columns and not rep.is_valid
    bad2 = raw_df.head(10).copy()
    bad2.loc[0, "BMI"] = 120
    rep2 = validate_dataframe(bad2, cfg.features, cfg.target, cfg["data"]["zero_as_missing"])
    assert rep2.out_of_range["BMI"] == 1


def test_split_is_stratified_disjoint_and_deterministic(raw_df, cfg):
    X_tr, X_te, y_tr, y_te = split_data(raw_df, cfg)
    assert len(X_tr) == 614 and len(X_te) == 154
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y_te.mean()) < 0.01
    X_tr2, *_ = split_data(raw_df, cfg)
    assert np.array_equal(X_tr.index, X_tr2.index)
