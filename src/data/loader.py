"""Data ingestion for the Pima Indians Diabetes dataset."""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


class DataIntegrityError(RuntimeError):
    """Raised when the dataset on disk does not match the pinned version."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(cfg: Config, force: bool = False) -> Path:
    """Download the raw CSV to ``data.raw_path`` and verify its checksum."""
    dest = cfg.path("data", "raw_path")
    if dest.exists() and not force:
        logger.info("Dataset already present at %s", dest)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset from %s", cfg["data"]["source_url"])
        urllib.request.urlretrieve(cfg["data"]["source_url"], dest)  # noqa: S310 (pinned https URL)
    verify_checksum(dest, cfg["data"].get("sha256"))
    return dest


def verify_checksum(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = sha256_of(path)
    if actual != expected:
        raise DataIntegrityError(
            f"Checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The dataset differs from the version used for the reported results."
        )


def load_raw(path: str | Path, verify_sha256: str | None = None) -> pd.DataFrame:
    """Load the dataset with canonical column names.

    Accepts both the header-less UCI/Brownlee CSV and the Kaggle CSV (with header row).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    if verify_sha256:
        verify_checksum(path, verify_sha256)

    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    has_header = any(ch.isalpha() for ch in first_line)
    if has_header:
        df = pd.read_csv(path)
        if len(df.columns) != len(CANONICAL_COLUMNS):
            raise ValueError(f"Expected {len(CANONICAL_COLUMNS)} columns, found {len(df.columns)}")
        df.columns = CANONICAL_COLUMNS
    else:
        df = pd.read_csv(path, header=None, names=CANONICAL_COLUMNS)
    return df


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the configured raw dataset, verifying the pinned checksum."""
    return load_raw(cfg.path("data", "raw_path"), verify_sha256=cfg["data"].get("sha256"))


def split_data(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with the project seed. No resampling happens here."""
    X = df[cfg.features]
    y = df[cfg.target].astype(int)
    return train_test_split(
        X,
        y,
        test_size=cfg["split"]["test_size"],
        stratify=y if cfg["split"]["stratify"] else None,
        random_state=cfg.seed,
    )


def save_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: Config,
) -> None:
    """Persist the split (with original row ids) so every later stage uses the same rows."""
    out = cfg.path("data", "processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{cfg.target: y_train}).to_csv(out / "train.csv", index_label="row_id")
    X_test.assign(**{cfg.target: y_test}).to_csv(out / "test.csv", index_label="row_id")


def load_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the persisted train/test split created by ``scripts/train.py``."""
    out = cfg.path("data", "processed_dir")
    train = pd.read_csv(out / "train.csv", index_col="row_id")
    test = pd.read_csv(out / "test.csv", index_col="row_id")
    return (
        train[cfg.features],
        test[cfg.features],
        train[cfg.target].astype(int),
        test[cfg.target].astype(int),
    )
