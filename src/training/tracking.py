"""Thin MLflow wrapper that degrades to a no-op when MLflow is disabled or unavailable."""

from __future__ import annotations

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


class ExperimentTracker:
    def __init__(self, cfg):
        self.enabled = bool(cfg["mlflow"].get("enabled", False))
        self._mlflow = None
        if not self.enabled:
            return
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow

            uri = cfg["mlflow"]["tracking_uri"]
            if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).is_absolute():
                # resolve relative SQLite paths against the project root, not the CWD
                uri = "sqlite:///" + (PROJECT_ROOT / uri[len("sqlite:///"):]).resolve().as_posix()
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
            self._mlflow = mlflow
        except Exception as exc:  # pragma: no cover - tracking must never break training
            logger.warning("MLflow disabled: %s", exc)
            self.enabled = False

    @contextlib.contextmanager
    def run(self, name: str, nested: bool = False, tags: dict | None = None):
        if not self.enabled:
            yield None
            return
        with self._mlflow.start_run(run_name=name, nested=nested, tags=tags) as r:
            yield r

    def log_params(self, params: dict) -> None:
        if self.enabled:
            flat = {k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in _flatten(params).items()}
            self._mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})

    def log_metrics(self, metrics: dict, prefix: str = "") -> None:
        if self.enabled:
            clean = {f"{prefix}{k}": float(v) for k, v in _flatten(metrics).items()
                     if isinstance(v, (int, float)) and v == v}
            self._mlflow.log_metrics(clean)

    def log_artifact(self, path: str | Path) -> None:
        if self.enabled and Path(path).exists():
            self._mlflow.log_artifact(str(path))
