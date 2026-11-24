"""Request/response schemas shared by the API, the predictor and the Streamlit app."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.data.validation import PLAUSIBLE_RANGES as R

# API field name -> canonical dataset column
FIELD_TO_COLUMN = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "skin_thickness": "SkinThickness",
    "insulin": "Insulin",
    "bmi": "BMI",
    "diabetes_pedigree": "DiabetesPedigreeFunction",
    "age": "Age",
}
COLUMN_TO_FIELD = {v: k for k, v in FIELD_TO_COLUMN.items()}

DISPLAY = {
    "Pregnancies": ("Pregnancies", "count"),
    "Glucose": ("Glucose (2-h OGTT)", "mg/dL"),
    "BloodPressure": ("Diastolic blood pressure", "mm Hg"),
    "SkinThickness": ("Triceps skinfold thickness", "mm"),
    "Insulin": ("2-hour serum insulin", "uU/mL"),
    "BMI": ("BMI", "kg/m^2"),
    "DiabetesPedigreeFunction": ("Diabetes pedigree function", "score"),
    "Age": ("Age", "years"),
}

ThresholdMode = Literal["balanced", "high_sensitivity", "high_precision"]


class PatientFeatures(BaseModel):
    """One patient's inputs. Physiologically impossible values are rejected (HTTP 422).

    ``skin_thickness`` and ``insulin`` are optional because they were unmeasured for a large
    share of the training data; omitted values are imputed with the training median and
    flagged in the response.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False,
                              json_schema_extra={"example": {
                                  "pregnancies": 2, "glucose": 140, "blood_pressure": 80, "skin_thickness": 25,
                                  "insulin": 100, "bmi": 30.5, "diabetes_pedigree": 0.45, "age": 40}})

    pregnancies: int = Field(..., ge=R["Pregnancies"][0], le=R["Pregnancies"][1], description="Number of pregnancies")
    glucose: float = Field(..., ge=R["Glucose"][0], le=R["Glucose"][1], description="2-hour OGTT plasma glucose, mg/dL")
    blood_pressure: float = Field(..., ge=R["BloodPressure"][0], le=R["BloodPressure"][1], description="Diastolic BP, mm Hg")
    skin_thickness: Optional[float] = Field(None, ge=R["SkinThickness"][0], le=R["SkinThickness"][1],
                                            description="Triceps skinfold, mm (optional)")
    insulin: Optional[float] = Field(None, ge=R["Insulin"][0], le=R["Insulin"][1],
                                     description="2-hour serum insulin, uU/mL (optional)")
    bmi: float = Field(..., ge=R["BMI"][0], le=R["BMI"][1], description="Body mass index, kg/m^2")
    diabetes_pedigree: float = Field(..., ge=R["DiabetesPedigreeFunction"][0], le=R["DiabetesPedigreeFunction"][1],
                                     description="Diabetes pedigree function")
    age: int = Field(..., ge=R["Age"][0], le=R["Age"][1], description="Age in years")

    def to_row(self) -> dict[str, float]:
        """Canonical-column dict; missing optional values become NaN (imputed by the model)."""
        return {col: (float("nan") if getattr(self, f) is None else float(getattr(self, f)))
                for f, col in FIELD_TO_COLUMN.items()}


class Contribution(BaseModel):
    feature: str
    display_name: str
    value: float
    unit: str
    imputed: bool
    shap_value: float
    direction: Literal["increases", "decreases", "neutral"]


class Explanation(BaseModel):
    method: str
    base_value: float
    contributions: list[Contribution]
    summary: str
    note: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    threshold_mode: str
    model_version: str
    risk_statement: str
    imputed_features: list[str]
    warnings: list[str]
    explanation: Optional[Explanation] = None
    disclaimer: str


class TextPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(..., min_length=3, max_length=1000)
    fill_missing_with_median: bool = True


class ParsedText(BaseModel):
    extracted: dict[str, Optional[float]]
    filled_with_training_median: list[str]
    missing_required: list[str]
    notes: list[str]
