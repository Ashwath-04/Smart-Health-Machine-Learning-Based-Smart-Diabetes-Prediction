"""Shared pipeline construction: preprocessing -> (optional) SMOTE -> classifier.

``imblearn.pipeline.Pipeline`` only calls the sampler's ``fit_resample`` during ``fit``.
At ``predict``/``predict_proba`` time the SMOTE step is skipped, so validation and test
rows are never oversampled. Because the whole chain is one estimator, cloning it inside
``cross_validate`` / ``StackingClassifier`` refits the imputer *and* SMOTE on each
training fold only.
"""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data.preprocessing import preprocessing_steps_from_config

SMOTE_K_PARAM = "smote_k_neighbors"


def build_pipeline(
    estimator: Any,
    cfg,
    smote: bool,
    scale: bool = False,
    smote_k_neighbors: int | None = None,
    seed: int | None = None,
) -> ImbPipeline:
    """Wrap ``estimator`` with leakage-safe preprocessing and optional SMOTE."""
    steps: list[tuple[str, Any]] = preprocessing_steps_from_config(cfg, scale=scale)
    if smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    k_neighbors=int(smote_k_neighbors or cfg["smote"]["k_neighbors"]),
                    sampling_strategy=cfg["smote"].get("sampling_strategy", "auto"),
                    random_state=cfg.seed if seed is None else seed,
                ),
            )
        )
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def split_smote_params(params: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
    """Separate the SMOTE ``k_neighbors`` (tuned jointly) from classifier params."""
    params = dict(params)
    k = params.pop(SMOTE_K_PARAM, None)
    return params, k
