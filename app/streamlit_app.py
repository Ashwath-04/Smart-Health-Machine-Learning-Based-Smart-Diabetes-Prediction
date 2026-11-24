"""Streamlit front-end.

Run:  streamlit run app/streamlit_app.py
If the environment variable API_URL is set (e.g. in docker-compose), predictions are requested
from the FastAPI service; otherwise the model is loaded in-process.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.evaluation import plots as P  # noqa: E402
from src.inference.explanation import DISCLAIMER, build_report  # noqa: E402
from src.inference.schemas import PatientFeatures  # noqa: E402

st.set_page_config(page_title="Smart Health - Diabetes Prediction", page_icon=":material/monitor_heart:", layout="wide")

CFG = load_config()
METRICS = CFG.path("paths", "metrics_dir")
FIGS = CFG.path("paths", "figures_dir")
API_URL = os.environ.get("API_URL", "").rstrip("/")
MODE_LABELS = {"balanced": "Balanced (max F1)", "high_sensitivity": "High sensitivity",
               "high_precision": "High precision"}


# ----------------------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Loading model...")
def local_predictor():
    from src.inference.predictor import DiabetesPredictor

    return DiabetesPredictor.from_config(CFG)


def api_call(method: str, path: str, **kwargs):
    import requests

    r = requests.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
    if r.status_code >= 400:
        raise RuntimeError(f"API error {r.status_code}: {r.text[:500]}")
    return r.json()


def predict(features: PatientFeatures, mode: str) -> dict:
    if API_URL:
        return api_call("POST", "/predict", json=features.model_dump(), params={"mode": mode, "explain": True})
    return local_predictor().predict(features, mode=mode, explain=True)


def model_info() -> dict:
    if API_URL:
        return api_call("GET", "/model-info")
    return local_predictor().info()


def parse_text(text: str) -> dict:
    if API_URL:
        return api_call("POST", "/parse", json={"text": text, "fill_missing_with_median": True})
    return local_predictor().parse(text, fill_missing=True)


@st.cache_data
def read_csv(name: str) -> pd.DataFrame | None:
    p = METRICS / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def read_json(name: str):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def figure(name: str, caption: str | None = None):
    p = FIGS / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")
    else:
        st.info(f"Figure `{name}` not found - run `python scripts/evaluate.py`.")


def disclaimer_banner():
    st.warning(f"**{DISCLAIMER}**", icon=":material/info:")


# ----------------------------------------------------------------------------- pages
def page_home():
    st.title("Smart Health: Diabetes Prediction")
    st.caption("Stacking ensemble - XGBoost + ExtraTrees + CatBoost -> Logistic Regression meta-learner")
    disclaimer_banner()
    st.markdown(
        """
This app estimates the probability that a patient in the **Pima Indians Diabetes** cohort profile
has diabetes (WHO 2-hour OGTT criterion), from eight routinely collected measurements.

**What is inside**
- Leakage-safe pipeline: median imputation of impossible zeros and SMOTE are fitted *inside* every training fold.
- Stacking ensemble trained on out-of-fold base-model predictions; Optuna-tuned base learners.
- Decision thresholds selected on cross-validated training predictions (never on the test set).
- SHAP explanations of every individual prediction.
- An honest comparison against single models and the no-SMOTE alternative (see *Model Performance*).
"""
    )
    dep = read_json("test_metrics_deployed.json")
    if dep:
        m = dep["metrics_by_threshold_mode"]["balanced"]
        c = st.columns(5)
        c[0].metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")
        c[1].metric("Test PR-AUC", f"{m['pr_auc']:.3f}")
        c[2].metric("Test recall", f"{m['recall']:.3f}")
        c[3].metric("Test precision", f"{m['precision']:.3f}")
        c[4].metric("Test F1", f"{m['f1']:.3f}")
        st.caption(f"Held-out test set: {dep['n_test']} patients, balanced threshold {m['threshold']:.2f}.")


def contributions_chart(contribs: list[dict]):
    import matplotlib.pyplot as plt

    d = pd.DataFrame(contribs).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    colors = [P.PALETTE[7] if v > 0 else P.PALETTE[0] for v in d["shap_value"]]
    labels = [f"{n} = {v:g}{' (imputed)' if imp else ''}" for n, v, imp in zip(d["display_name"], d["value"], d["imputed"])]
    ax.barh(labels, d["shap_value"], color=colors, height=0.6)
    ax.axvline(0, color=P.AXIS, lw=1)
    for i, v in enumerate(d["shap_value"]):
        ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=P.INK_2)
    lim = max(abs(d["shap_value"]).max() * 1.35, 0.02)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Contribution to predicted probability (red raises, blue lowers)")
    ax.grid(axis="y", visible=False)
    st.pyplot(fig, width="stretch")


def probability_bar(prob: float, thresholds: dict, mode: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh([0], [1], color=P.GRID, height=0.5)
    ax.barh([0], [prob], color=P.PALETTE[1] if prob >= thresholds[mode] else P.PALETTE[0], height=0.5)
    for m, t in thresholds.items():
        ax.axvline(t, color=P.INK if m == mode else P.MUTED, lw=1.6 if m == mode else 0.8, ls="-" if m == mode else ":")
        ax.text(t, 0.42, m.replace("_", " "), fontsize=7, ha="center", color=P.INK_2)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Predicted probability")
    ax.grid(False)
    st.pyplot(fig, width="stretch")


def page_predict():
    st.title("Diabetes risk prediction")
    disclaimer_banner()
    try:
        info = model_info()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Model unavailable: {exc}")
        return
    thresholds = info["thresholds"]

    with st.expander("Describe the patient in plain English (optional)", icon=":material/edit_note:"):
        text = st.text_input("Example: 45-year-old with glucose 160, BMI 31 and blood pressure 85",
                             key="nl_text")
        if st.button("Fill the form from text") and text:
            parsed = parse_text(text)
            for k, v in parsed["extracted"].items():
                if v is not None:
                    st.session_state[f"f_{k}"] = int(v) if k in {"pregnancies", "age"} else float(v)
            for k in ("skin_thickness", "insulin"):
                st.session_state[f"has_{k}"] = parsed["extracted"].get(k) is not None
            for n in parsed["notes"]:
                st.info(n)

    defaults = {"pregnancies": 2, "glucose": 140.0, "blood_pressure": 80.0, "skin_thickness": 25.0, "insulin": 100.0,
                "bmi": 30.5, "diabetes_pedigree": 0.45, "age": 40}
    for k, v in defaults.items():
        st.session_state.setdefault(f"f_{k}", v)
    st.session_state.setdefault("has_skin_thickness", True)
    st.session_state.setdefault("has_insulin", True)

    with st.form("patient"):
        c1, c2, c3, c4 = st.columns(4)
        preg = c1.number_input("Pregnancies", 0, 20, step=1, key="f_pregnancies")
        glu = c2.number_input("Glucose (mg/dL, 2-h OGTT)", 1.0, 600.0, step=1.0, key="f_glucose")
        bp = c3.number_input("Diastolic blood pressure (mm Hg)", 1.0, 200.0, step=1.0, key="f_blood_pressure")
        bmi = c4.number_input("BMI (kg/m²)", 10.0, 80.0, step=0.1, key="f_bmi")
        c5, c6, c7, c8 = st.columns(4)
        dpf = c5.number_input("Diabetes pedigree function", 0.0, 3.0, step=0.01, format="%.3f", key="f_diabetes_pedigree")
        age = c6.number_input("Age (years)", 18, 120, step=1, key="f_age")
        has_skin = c7.checkbox("Skin thickness measured", key="has_skin_thickness")
        skin = c7.number_input("Triceps skinfold (mm)", 1.0, 100.0, step=1.0, key="f_skin_thickness", disabled=not has_skin)
        has_ins = c8.checkbox("Insulin measured", key="has_insulin")
        ins = c8.number_input("2-h serum insulin (µU/mL)", 1.0, 1000.0, step=1.0, key="f_insulin", disabled=not has_ins)
        mode = st.radio("Operating mode", list(MODE_LABELS), format_func=lambda m: f"{MODE_LABELS[m]} (threshold {thresholds[m]:.2f})",
                        horizontal=True)
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return
    try:
        features = PatientFeatures(pregnancies=int(preg), glucose=glu, blood_pressure=bp, bmi=bmi,
                                   diabetes_pedigree=dpf, age=int(age), skin_thickness=skin if has_skin else None,
                                   insulin=ins if has_ins else None)
        with st.spinner("Scoring and computing SHAP explanation..."):
            res = predict(features, mode)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader(res["risk_statement"])
    a, b, c = st.columns(3)
    a.metric("Predicted probability", f"{res['probability']:.3f}")
    b.metric("Threshold", f"{res['threshold']:.2f}", help=MODE_LABELS[res["threshold_mode"]])
    c.metric("Model class", "1 - higher likelihood" if res["prediction"] else "0 - lower likelihood")
    probability_bar(res["probability"], thresholds, mode)
    for w in res["warnings"]:
        st.warning(w)
    if res["imputed_features"]:
        st.info("Not supplied, imputed with the training median: " + ", ".join(res["imputed_features"]))
    exp = res.get("explanation")
    if exp:
        st.markdown("#### Why the model produced this estimate")
        contributions_chart(exp["contributions"])
        st.write(exp["summary"])
        st.caption(exp["note"])
        st.download_button("Download model report (Markdown)", build_report(res, info), file_name="model_report.md")
    st.caption(f"Model version: {res['model_version']}")


def page_performance():
    st.title("Model performance")
    disclaimer_banner()
    final = read_csv("final_comparison_table.csv")
    cv = read_csv("cv_summary.csv")
    if final is None or cv is None:
        st.info("Run `python scripts/train.py` and `python scripts/evaluate.py` first.")
        return
    sel = read_json("model_selection.json")
    st.markdown(f"**Deployed model:** `{sel['deployed_model']}` - **best single candidate by CV {sel['selection_metric']}:** "
                f"`{sel['best_by_cv']}`. {sel['note']}")

    tab_table, tab_curves, tab_smote, tab_thr, tab_cv = st.tabs(
        ["Comparison table", "Curves & confusion matrices", "SMOTE vs no SMOTE", "Threshold explorer", "CV stability"])

    with tab_table:
        f1, f2, f3 = st.columns(3)
        fams = f1.multiselect("Model family", sorted(cv.family.unique()), default=sorted(cv.family.unique()))
        smote_opt = f2.selectbox("SMOTE", ["both", "SMOTE only", "no SMOTE only"])
        tuned_opt = f3.selectbox("Tuning", ["both", "tuned only", "default only"])
        mask = cv.family.isin(fams)
        if smote_opt != "both":
            mask &= cv.smote == (smote_opt == "SMOTE only")
        if tuned_opt != "both":
            mask &= cv.tuned == (tuned_opt == "tuned only")
        keep = cv.loc[mask, "candidate"]
        st.markdown("**Held-out test set** (each model at its own CV-selected balanced threshold)")
        st.dataframe(final[final.candidate.isin(keep)].drop(columns=["display_name"]).round(3), hide_index=True,
                     width="stretch")
        st.markdown("**Cross-validation on training set** (3x5-fold; threshold-dependent metrics at 0.5)")
        cols = ["candidate", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "f1_mean", "f1_std", "recall_mean",
                "precision_mean", "brier_mean", "calibration", "threshold_balanced", "oof_f1_at_balanced"]
        st.dataframe(cv.loc[mask, cols].sort_values("pr_auc_mean", ascending=False).round(4), hide_index=True,
                     width="stretch")

    with tab_curves:
        preds = read_csv("test_predictions.csv")
        options = [c for c in preds.columns if c not in ("row_id", "y_true")]
        default = [c for c in [sel["deployed_model"], "xgboost_smote_tuned", "extratrees_smote_tuned",
                               "catboost_smote_tuned"] if c in options]
        chosen = st.multiselect("Models to compare (test set)", options, default=default, max_selections=6)
        if chosen:
            labels = {c: c for c in chosen}
            curves = {c: (preds.y_true.to_numpy(), preds[c].to_numpy()) for c in chosen}
            g1, g2, g3 = st.columns(3)
            g1.pyplot(P.plot_roc_curves(curves, labels), width="stretch")
            g2.pyplot(P.plot_pr_curves(curves, labels), width="stretch")
            g3.pyplot(P.plot_calibration_curves(curves, labels), width="stretch")
            thr = final.set_index("candidate")["threshold"]
            cms = {c: (preds.y_true.to_numpy(), (preds[c] >= thr[c]).astype(int).to_numpy()) for c in chosen}
            st.pyplot(P.plot_confusion_matrices(cms, {c: f"{c}\nthr={thr[c]:.2f}" for c in chosen}), width="stretch")
        figure("test_f1_comparison.png")
        figure("test_recall_comparison.png")

    with tab_smote:
        sm = read_csv("smote_comparison.csv")
        eff = read_csv("smote_effect.csv")
        if sm is not None:
            st.dataframe(sm.round(3), hide_index=True, width="stretch")
            st.markdown("**SMOTE effect (SMOTE minus no-SMOTE)** - `pp` = percentage points")
            st.dataframe(eff.round(3), hide_index=True, width="stretch")
        figure("cv_recall_comparison.png")
        figure("cv_pr_auc_comparison.png")

    with tab_thr:
        dep = sel["deployed_model"]
        tt = read_csv(f"threshold_table_{dep}.csv")
        st.markdown("Threshold sweep on **training out-of-fold predictions** (the data used to choose thresholds).")
        t = st.slider("Threshold", float(tt.threshold.min()), float(tt.threshold.max()), 0.5, 0.01)
        row = tt.iloc[(tt.threshold - t).abs().argmin()]
        c = st.columns(7)
        for col, k in zip(c, ["tp", "fn", "fp", "tn", "precision", "recall", "f1"]):
            counts = k in ("tp", "fn", "fp", "tn")
            col.metric(k.upper() if counts else k.capitalize(), int(row[k]) if counts else f"{row[k]:.3f}")
        st.caption("Counts are pooled over 3 CV repeats (each training patient appears 3 times).")
        info = model_info()
        st.pyplot(P.plot_threshold_tradeoff(tt, info["thresholds"]), width="stretch")

    with tab_cv:
        figure("cv_boxplot_pr_auc.png")
        figure("cv_boxplot_f1.png")
        stab = read_csv("stability_summary.csv")
        if stab is not None:
            st.markdown("**Seed sensitivity** (5 seeds x 5-fold CV; split and model seeds varied)")
            st.dataframe(stab.round(4), hide_index=True, width="stretch")


def page_explain():
    st.title("Explainability")
    disclaimer_banner()
    st.markdown("SHAP values below are **model contributions**: how each feature moved the deployed model's predicted "
                "probability. They are not evidence that a feature causes diabetes.")
    t1, t2, t3, t4 = st.tabs(["Global SHAP", "Stacking internals", "Error analysis", "Individual examples"])
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            figure("shap_summary_beeswarm.png")
        with c2:
            figure("shap_bar.png")
        gi = read_csv("shap_global_importance.csv")
        if gi is not None:
            st.dataframe(gi.round(4), hide_index=True)
        figure("permutation_importance.png", "Permutation importance: drop in test PR-AUC when a feature is shuffled")
        c1, c2, c3 = st.columns(3)
        for c, f in zip((c1, c2, c3), ["Glucose", "BMI", "Age"]):
            with c:
                figure(f"shap_dependence_{f}.png")
    with t2:
        sa = read_json("stacking_analysis.json")
        if sa:
            st.markdown("Meta-learner (logistic regression) weights on each base model's out-of-fold probability:")
            st.dataframe(pd.DataFrame({"coefficient": sa["meta_coefficients"],
                                       "standardized": sa["meta_standardized_coefficients"]}).round(3))
            st.markdown("Correlation of base-model OOF probabilities (high = little diversity):")
            st.dataframe(pd.DataFrame(sa["oof_correlation"]).round(3))
            st.markdown("Drop-one ablation of the meta-learner (CV on OOF meta-features):")
            st.dataframe(pd.DataFrame(sa["drop_one"]).T.round(4))
        nat = read_csv("native_importances_base_learners.csv")
        if nat is not None:
            st.markdown("Native importances inside each base learner (normalised):")
            st.dataframe(nat.round(3), hide_index=True)
    with t3:
        figure("errors_test_features.png")
        figure("errors_oof_features.png")
        figure("errors_test_probability.png")
        es = read_json("error_analysis_summary.json")
        if es:
            st.json(es)
    with t4:
        for et, desc in [("TP", "diabetic, predicted positive"), ("FN", "diabetic, predicted negative (missed)"),
                         ("FP", "non-diabetic, predicted positive")]:
            st.markdown(f"**{et}** - {desc}")
            figure(f"shap_waterfall_{et}.png")


def page_about():
    st.title("About the model")
    disclaimer_banner()
    try:
        info = model_info()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    c = st.columns(3)
    c[0].metric("Model", info.get("model_name", "?"))
    c[1].metric("Calibration", info.get("calibration", {}).get("method", "?"))
    c[2].metric("Trained", str(info.get("training_date", "?"))[:10])
    st.markdown(
        """
**Pipeline (per base learner, refitted inside every CV fold):**
`zeros -> NaN (Glucose, BloodPressure, SkinThickness, Insulin, BMI)` -> `median imputation` -> `SMOTE (training folds only)` -> `classifier`

**Stacking:** XGBoost, ExtraTrees and CatBoost produce out-of-fold probabilities (5-fold, inner CV) that train a
logistic-regression meta-learner; base learners are then refitted on all training data.

**Intended use:** education and research on ensemble learning and imbalanced classification.
**Not intended for:** diagnosis, screening decisions, or any clinical use.

**Known limitations:** small dataset (768 patients; 154 in test), a single population (Pima Indian women aged 21+),
historical data, many missing insulin/skin-thickness values, and no external validation.
"""
    )
    st.subheader("Model metadata")
    st.json(info, expanded=False)


PAGES = {
    "Home": page_home,
    "Diabetes Prediction": page_predict,
    "Model Performance": page_performance,
    "Explainability": page_explain,
    "About Model": page_about,
}

with st.sidebar:
    st.header("Smart Health")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.caption("Educational/research prototype. Not a medical device.")
    st.caption(f"Backend: {'API ' + API_URL if API_URL else 'local model'}")
PAGES[page]()
