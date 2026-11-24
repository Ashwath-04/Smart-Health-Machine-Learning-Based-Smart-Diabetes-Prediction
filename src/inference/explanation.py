"""Grounded, template-based explanation and report generation.

Everything in the generated text is derived from the model output and its SHAP values; no
medical knowledge is generated. This deliberately avoids an LLM: a deterministic template
cannot invent clinical facts, needs no API key and is fully testable. (An optional LLM
re-phrasing layer is discussed as future work in the README.)
"""

from __future__ import annotations

from src.inference.schemas import DISPLAY

DISCLAIMER = (
    "This application is an educational/research prototype and is not a substitute for "
    "professional medical evaluation."
)
SHAP_NOTE = (
    "Contributions show how each input moved this model's predicted probability relative to its "
    "average prediction on training patients. They describe the model's behaviour, not clinical "
    "causation, and must not be read as medical advice."
)


def risk_statement(prediction: int) -> str:
    if prediction == 1:
        return "The model predicts a higher likelihood of diabetes based on the supplied features."
    return "The model predicts a lower likelihood of diabetes based on the supplied features."


def _fmt_value(feature: str, value: float) -> str:
    unit = DISPLAY[feature][1]
    v = f"{value:.0f}" if feature in {"Pregnancies", "Age", "Glucose", "BloodPressure", "Insulin", "SkinThickness"} else f"{value:.2f}" if feature == "DiabetesPedigreeFunction" else f"{value:.1f}"
    return v if unit in {"count", "score"} else f"{v} {unit}"


def summarize(probability: float, threshold: float, mode: str, contributions: list[dict], top_k: int = 3) -> str:
    """Plain-language summary of the model output and its largest SHAP contributions."""
    side = "above" if probability >= threshold else "below"
    parts = [
        f"The model's estimated probability is {probability:.2f}, which is {side} the "
        f"{mode.replace('_', ' ')} threshold of {threshold:.2f}."
    ]
    up = [c for c in contributions if c["shap_value"] > 0][:top_k]
    down = [c for c in contributions if c["shap_value"] < 0][:top_k]

    def phrase(c):
        tag = " (imputed - not supplied)" if c.get("imputed") else ""
        return f"{DISPLAY[c['feature']][0]} = {_fmt_value(c['feature'], c['value'])}{tag} ({c['shap_value']:+.3f})"

    if up:
        parts.append("Inputs that raised the model's estimate the most: " + "; ".join(phrase(c) for c in up) + ".")
    if down:
        parts.append("Inputs that lowered it the most: " + "; ".join(phrase(c) for c in down) + ".")
    parts.append("These are model contributions, not causes.")
    return " ".join(parts)


def build_report(result: dict, model_info: dict) -> str:
    """Markdown patient-level model report (prediction, probability, contributions, limitations)."""
    lines = [
        "# Model output report",
        "",
        f"- **Model:** {model_info.get('display_name', model_info.get('model_name'))} "
        f"(version `{result['model_version']}`)",
        f"- **Model output:** {result['risk_statement']}",
        f"- **Predicted probability:** {result['probability']:.3f}",
        f"- **Decision threshold:** {result['threshold']:.2f} ({result['threshold_mode'].replace('_', ' ')} mode)",
    ]
    if result["imputed_features"]:
        lines.append(f"- **Not supplied (imputed with training median):** {', '.join(result['imputed_features'])}")
    for w in result["warnings"]:
        lines.append(f"- **Warning:** {w}")
    exp = result.get("explanation")
    if exp:
        lines += ["", "## Feature contributions (SHAP, probability units)", "",
                  "| Input | Value | Contribution | Direction |", "|---|---|---|---|"]
        for c in exp["contributions"]:
            lines.append(f"| {c['display_name']} | {_fmt_value(c['feature'], c['value'])}"
                         f"{' (imputed)' if c['imputed'] else ''} | {c['shap_value']:+.3f} | {c['direction']} |")
        lines += ["", exp["summary"], "", f"_{exp['note']}_"]
    lines += [
        "",
        "## Limitations",
        "",
        "- Trained on 768 records of Pima Indian women aged 21+ (1990s data); it may not generalise to other populations.",
        "- Performance was estimated on 154 held-out patients; confidence intervals are wide.",
        "- Not clinically validated. Not a diagnostic device.",
        "",
        f"**{DISCLAIMER}**",
    ]
    return "\n".join(lines)
