"""Configuration loading.

All tunable values live in ``configs/config.yaml``. This module only resolves paths and
exposes the configuration as a nested dictionary with a few typed helpers.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """Dictionary-backed configuration with path helpers."""

    @property
    def seed(self) -> int:
        return int(self["project"]["random_seed"])

    @property
    def features(self) -> list[str]:
        return list(self["data"]["features"])

    @property
    def target(self) -> str:
        return str(self["data"]["target"])

    def path(self, *keys: str) -> Path:
        """Resolve a config value (relative to the project root) to an absolute path."""
        node: Any = self
        for key in keys:
            node = node[key]
        p = Path(node)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def copy_with(self, overrides: dict[str, Any]) -> "Config":
        """Return a deep copy with nested ``overrides`` merged in."""
        new = Config(copy.deepcopy(dict(self)))
        _deep_update(new, overrides)
        return new


def _deep_update(base: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the YAML configuration (``DIABETES_CONFIG`` env var overrides the default path)."""
    cfg_path = Path(path or os.environ.get("DIABETES_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def set_global_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs. Model-level seeds are passed explicitly to estimators."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
