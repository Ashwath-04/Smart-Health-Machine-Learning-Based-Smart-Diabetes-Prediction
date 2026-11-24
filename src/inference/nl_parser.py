"""Rule-based natural-language -> structured input parser.

Example: "Predict diabetes risk for a 45-year-old with glucose 160 and BMI 31".
The parser only extracts numbers the user actually wrote. It never guesses a value: fields
that are not mentioned are either reported as missing or (if the caller allows it) filled
with the training median and explicitly flagged.
"""

from __future__ import annotations

import re

NUM = r"(\d+(?:\.\d+)?)"
SEP = r"\s*(?:of|is|was|at|=|:|level|levels|reading)?\s*"

PATTERNS: dict[str, list[str]] = {
    "age": [rf"{NUM}\s*-?\s*(?:years?|yrs?|y/?o)\b(?:\s*-?\s*old)?", rf"\bage(?:d)?{SEP}{NUM}"],
    "glucose": [rf"\b(?:glucose|blood sugar|sugar|plasma glucose|ogtt){SEP}{NUM}"],
    "bmi": [rf"\b(?:bmi|body mass index){SEP}{NUM}"],
    "blood_pressure": [rf"\b(?:blood pressure|bp){SEP}\d+\s*/\s*{NUM}",
                       rf"\bdiastolic(?: blood pressure| bp)?{SEP}{NUM}",
                       rf"\b(?:blood pressure|bp){SEP}{NUM}"],
    "insulin": [rf"\binsulin{SEP}{NUM}"],
    "skin_thickness": [rf"\b(?:skin ?fold|skin thickness|triceps(?: skinfold)?){SEP}{NUM}"],
    "diabetes_pedigree": [rf"\b(?:diabetes pedigree(?: function)?|pedigree(?: function)?|dpf){SEP}{NUM}"],
    "pregnancies": [rf"{NUM}\s*(?:pregnancies|pregnancy|times pregnant)", rf"\bpregnancies{SEP}{NUM}"],
}
ZERO_PREGNANCY = re.compile(r"\b(?:no|zero|never)\s+(?:been\s+)?(?:pregnancies|pregnant)\b")
REQUIRED = ["pregnancies", "glucose", "blood_pressure", "bmi", "diabetes_pedigree", "age"]


def parse_text(text: str) -> tuple[dict[str, float | None], list[str]]:
    """Return (extracted values, notes). Unmentioned fields map to ``None``."""
    t = text.lower()
    values: dict[str, float | None] = {k: None for k in PATTERNS}
    notes: list[str] = []
    for field, pats in PATTERNS.items():
        for pat in pats:
            m = re.search(pat, t)
            if m:
                values[field] = float(m.group(1))
                if field == "blood_pressure" and "/" in m.group(0):
                    notes.append("Blood pressure given as systolic/diastolic; the diastolic value was used.")
                break
    if values["pregnancies"] is None and ZERO_PREGNANCY.search(t):
        values["pregnancies"] = 0.0
    return values, notes


def complete_with_medians(values: dict[str, float | None], medians_by_field: dict[str, float],
                          fill: bool) -> tuple[dict[str, float | None], list[str], list[str]]:
    """Fill missing *required* fields with training medians (flagged), or report them."""
    out = dict(values)
    filled, missing = [], []
    for f in REQUIRED:
        if out.get(f) is None:
            if fill:
                v = medians_by_field[f]
                out[f] = round(v) if f in {"pregnancies", "age"} else v
                filled.append(f)
            else:
                missing.append(f)
    return out, filled, missing
