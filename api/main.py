"""FastAPI prediction service.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import __version__  # noqa: E402
from src.inference.explanation import DISCLAIMER, build_report  # noqa: E402
from src.inference.predictor import DiabetesPredictor, ModelNotLoadedError  # noqa: E402
from src.inference.schemas import (  # noqa: E402
    ParsedText,
    PatientFeatures,
    PredictionResponse,
    TextPredictionRequest,
    ThresholdMode,
)

logger = logging.getLogger("api")
STATE: dict[str, Any] = {"predictor": None, "load_error": None}


def load_predictor() -> None:
    try:
        STATE["predictor"] = DiabetesPredictor.from_config()
        STATE["load_error"] = None
        logger.info("Loaded model %s", STATE["predictor"].model_version)
    except (ModelNotLoadedError, FileNotFoundError, TypeError) as exc:
        STATE["predictor"], STATE["load_error"] = None, str(exc)
        logger.error("Model not loaded: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_predictor()
    if STATE["predictor"] is not None and os.environ.get("WARM_SHAP", "1") == "1":
        STATE["predictor"].warmup()  # build the SHAP explainer + JIT once at start-up
    yield


app = FastAPI(
    title="Smart Health - Diabetes Prediction API",
    description=f"Stacking ensemble (XGBoost + ExtraTrees + CatBoost -> Logistic Regression). {DISCLAIMER}",
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": errors})


def predictor() -> DiabetesPredictor:
    if STATE["predictor"] is None:
        raise HTTPException(status_code=503, detail=f"Model not available: {STATE['load_error']}")
    return STATE["predictor"]


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Opening the base URL in a browser lands on the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, Any]:
    p = STATE["predictor"]
    return {"status": "ok" if p else "degraded", "model_loaded": p is not None,
            "model_version": p.model_version if p else None, "error": STATE["load_error"]}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {**predictor().info(), "disclaimer": DISCLAIMER}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures,
            mode: ThresholdMode = Query("balanced", description="Operating point"),
            explain: bool = Query(True, description="Include SHAP explanation")) -> dict[str, Any]:
    return predictor().predict(features, mode=mode, explain=explain)


@app.post("/parse", response_model=ParsedText)
def parse(req: TextPredictionRequest) -> dict[str, Any]:
    """Convert free text into structured model inputs (no prediction)."""
    return predictor().parse(req.text, fill_missing=req.fill_missing_with_median)


@app.post("/predict/text")
def predict_text(req: TextPredictionRequest, mode: ThresholdMode = Query("balanced")) -> dict[str, Any]:
    p = predictor()
    parsed = p.parse(req.text, fill_missing=req.fill_missing_with_median)
    if parsed["missing_required"]:
        raise HTTPException(status_code=422, detail={"message": "Required inputs not found in text",
                                                     "missing": parsed["missing_required"], "parsed": parsed})
    try:
        features = p.features_from_parsed(parsed)
    except ValueError as exc:  # pydantic ValidationError subclasses ValueError
        raise HTTPException(status_code=422, detail={"message": "Parsed values are out of range",
                                                     "error": str(exc), "parsed": parsed}) from exc
    return {"parsed": parsed, "result": p.predict(features, mode=mode, explain=True)}


@app.post("/report", response_class=PlainTextResponse)
def report(features: PatientFeatures, mode: ThresholdMode = Query("balanced")) -> str:
    """Markdown model-output report for one patient."""
    p = predictor()
    return build_report(p.predict(features, mode=mode, explain=True), p.info())
