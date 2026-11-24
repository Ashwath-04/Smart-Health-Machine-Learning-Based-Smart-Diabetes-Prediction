"""Final, one-time evaluation on the untouched test set + research analysis.

Every threshold, calibration choice and hyperparameter used here was fixed during training
from cross-validation on the training set. This module only *measures*; it never selects.
Each invocation is appended to ``reports/metrics/test_evaluation_log.jsonl`` so the number
of times the test set was scored is auditable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, roc_auc_score

from src.config import Config, set_global_seed
from src.data.loader import load_split, sha256_of
from src.evaluation import error_analysis as ea
from src.evaluation.metrics import (
    bootstrap_ci,
    compute_metrics,
    paired_bootstrap_diff,
    thresholded,
)
from src.evaluation.plots import (
    plot_calibration_curves,
    plot_confusion_matrices,
    plot_cv_boxplots,
    plot_metric_bars,
    plot_pr_curves,
    plot_probability_histogram,
    plot_roc_curves,
    plot_threshold_tradeoff,
)
from src.explainability.feature_importance import native_importances, permutation_importance_ap, plot_importance_bars
from src.explainability.shap_analysis import (
    ShapExplainer,
    background_sample,
    contributions,
    global_importance,
    impute_like_training,
    save_global_plots,
    save_waterfall,
)
from src.models.bundle import ModelBundle
from src.training.tracking import ExperimentTracker
from src.training.train import write_json

logger = logging.getLogger(__name__)

SHORT = {
    "logistic_regression": "LogReg",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "extratrees": "ExtraTrees",
    "catboost": "CatBoost",
    "stacking": "Stacking",
    "soft_voting": "SoftVoting",
}


def short_label(name: str) -> str:
    fam = next((f for f in SHORT if name.startswith(f)), name)
    rest = name[len(fam) + 1:].split("_")
    smote = "SMOTE" if rest[0] == "smote" else "no SMOTE"
    return f"{SHORT.get(fam, fam)} ({smote}, {rest[1]})"


def _pct(new: float, old: float) -> float | None:
    return None if old == 0 else 100.0 * (new - old) / old


def run_evaluation(cfg: Config, run_shap: bool = True) -> dict:
    set_global_seed(cfg.seed)
    metrics_dir, figures_dir, models_dir = (cfg.path("paths", "metrics_dir"), cfg.path("paths", "figures_dir"),
                                           cfg.path("paths", "models_dir"))
    X_train, X_test, y_train, y_test = load_split(cfg)
    selection = json.loads((metrics_dir / "model_selection.json").read_text())
    deployed = selection["deployed_model"]
    cv_summary = pd.read_csv(metrics_dir / "cv_summary.csv")
    folds = pd.read_csv(metrics_dir / "cv_fold_metrics.csv")
    oof = pd.read_csv(metrics_dir / "cv_oof_predictions.csv")
    zero_cols = cfg["data"]["zero_as_missing"]
    B = cfg["evaluation"]["bootstrap_resamples"]

    # ---------------- 1. score every candidate on the test set (once) ----------------
    bundles: dict[str, ModelBundle] = {}
    probas: dict[str, np.ndarray] = {}
    rows = []
    for path in sorted((models_dir / "candidates").glob("*.joblib")):
        b = ModelBundle.load(path)
        bundles[b.name] = b
        p = b.predict_proba(X_test)
        probas[b.name] = p
        for mode, thr in [("0.5", 0.5), ("balanced", b.thresholds["balanced"])]:
            rows.append({"candidate": b.name, "display_name": b.metadata["display_name"], "threshold_mode": mode,
                         "calibration": b.calibrator.method, **compute_metrics(y_test, p, thr)})
    test_all = pd.DataFrame(rows)
    test_all.to_csv(metrics_dir / "test_metrics_all.csv", index=False)
    pd.DataFrame({"row_id": X_test.index, "y_true": y_test.to_numpy(), **probas}).to_csv(
        metrics_dir / "test_predictions.csv", index=False)

    dep = bundles[deployed]
    p_dep = probas[deployed]
    dep_modes = {mode: compute_metrics(y_test, p_dep, thr) for mode, thr in dep.thresholds.items()}
    dep_modes["default_0.5"] = compute_metrics(y_test, p_dep, 0.5)
    thr_bal = dep.thresholds["balanced"]
    ci = {
        "f1": bootstrap_ci(y_test, p_dep, thresholded("f1", thr_bal), B, cfg.seed),
        "recall": bootstrap_ci(y_test, p_dep, thresholded("recall", thr_bal), B, cfg.seed),
        "precision": bootstrap_ci(y_test, p_dep, thresholded("precision", thr_bal), B, cfg.seed),
        "roc_auc": bootstrap_ci(y_test, p_dep, roc_auc_score, B, cfg.seed),
        "pr_auc": bootstrap_ci(y_test, p_dep, average_precision_score, B, cfg.seed),
        "brier": bootstrap_ci(y_test, p_dep, brier_score_loss, B, cfg.seed),
    }
    raw_dep = dep.model.predict_proba(X_test[dep.features])[:, 1]
    report_txt = classification_report(y_test, (p_dep >= thr_bal).astype(int),
                                       target_names=["Non-diabetic", "Diabetic"], digits=3)
    (metrics_dir / "classification_report_deployed.txt").write_text(
        f"Model: {deployed}\nThreshold (balanced, chosen on training OOF): {thr_bal}\n\n{report_txt}")
    deployed_test = {
        "model": deployed,
        "n_test": int(len(y_test)),
        "test_prevalence": float(y_test.mean()),
        "metrics_by_threshold_mode": dep_modes,
        "thresholds": dep.thresholds,
        "bootstrap_95ci_at_balanced": {k: {"point": v[0], "low": v[1], "high": v[2]} for k, v in ci.items()},
        "calibration": {
            "method": dep.calibrator.method,
            "brier_raw": float(brier_score_loss(y_test, raw_dep)),
            "brier_calibrated": float(brier_score_loss(y_test, p_dep)),
        },
    }
    write_json(deployed_test, metrics_dir / "test_metrics_deployed.json")

    # ---------------- 2. final comparison table ----------------
    final_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "sensitivity", "brier"]
    final = test_all[test_all.threshold_mode == "balanced"][["candidate", "display_name", "threshold"] + final_cols]
    final = final.assign(selected=final.candidate == deployed).sort_values("pr_auc", ascending=False)
    final.to_csv(metrics_dir / "final_comparison_table.csv", index=False)

    # ---------------- 3. research questions / claim verification ----------------
    cvs = cv_summary.set_index("candidate")
    t05 = test_all[test_all.threshold_mode == "0.5"].set_index("candidate")
    tbal = test_all[test_all.threshold_mode == "balanced"].set_index("candidate")
    smote_rows = []
    for fam, tag in [("logistic_regression", "default"), ("random_forest", "default"), ("xgboost", "tuned"),
                     ("extratrees", "tuned"), ("catboost", "tuned"), ("stacking", "tuned"),
                     ("xgboost", "default"), ("extratrees", "default"), ("catboost", "default"),
                     ("stacking", "default")]:
        a, b = f"{fam}_nosmote_{tag}", f"{fam}_smote_{tag}"
        if a not in cvs.index or b not in cvs.index:
            continue
        for smote, name in [(False, a), (True, b)]:
            smote_rows.append({
                "model": f"{SHORT[fam]} ({tag})", "smote": smote,
                "cv_precision@0.5": cvs.loc[name, "precision_mean"], "cv_recall@0.5": cvs.loc[name, "recall_mean"],
                "cv_f1@0.5": cvs.loc[name, "f1_mean"], "cv_roc_auc": cvs.loc[name, "roc_auc_mean"],
                "cv_pr_auc": cvs.loc[name, "pr_auc_mean"], "cv_brier": cvs.loc[name, "brier_mean"],
                "test_precision@0.5": t05.loc[name, "precision"], "test_recall@0.5": t05.loc[name, "recall"],
                "test_f1@0.5": t05.loc[name, "f1"], "test_roc_auc": t05.loc[name, "roc_auc"],
                "test_pr_auc": t05.loc[name, "pr_auc"],
                "tuned_threshold": tbal.loc[name, "threshold"],
                "test_recall@tuned": tbal.loc[name, "recall"], "test_f1@tuned": tbal.loc[name, "f1"],
            })
    smote_table = pd.DataFrame(smote_rows)
    smote_table.to_csv(metrics_dir / "smote_comparison.csv", index=False)

    smote_effect = []
    for model, g in smote_table.groupby("model", sort=False):
        off, on = g[~g.smote].iloc[0], g[g.smote].iloc[0]
        smote_effect.append({
            "model": model,
            "cv_recall@0.5_delta_pp": 100 * (on["cv_recall@0.5"] - off["cv_recall@0.5"]),
            "cv_recall@0.5_rel_pct": _pct(on["cv_recall@0.5"], off["cv_recall@0.5"]),
            "cv_f1@0.5_delta_pp": 100 * (on["cv_f1@0.5"] - off["cv_f1@0.5"]),
            "cv_pr_auc_delta": on["cv_pr_auc"] - off["cv_pr_auc"],
            "cv_roc_auc_delta": on["cv_roc_auc"] - off["cv_roc_auc"],
            "cv_brier_delta": on["cv_brier"] - off["cv_brier"],
            "test_recall@0.5_delta_pp": 100 * (on["test_recall@0.5"] - off["test_recall@0.5"]),
            "test_f1@0.5_delta_pp": 100 * (on["test_f1@0.5"] - off["test_f1@0.5"]),
            "test_f1@tuned_delta_pp": 100 * (on["test_f1@tuned"] - off["test_f1@tuned"]),
        })
    smote_effect = pd.DataFrame(smote_effect)
    smote_effect.to_csv(metrics_dir / "smote_effect.csv", index=False)

    # Stacking vs individual learners (same SMOTE + tuned configuration)
    stack = "stacking_smote_tuned"
    singles = ["xgboost_smote_tuned", "extratrees_smote_tuned", "catboost_smote_tuned"]
    stack_vs = []
    if stack in probas:
        for s in singles + ["soft_voting_smote_tuned", "logistic_regression_nosmote_default"]:
            if s not in probas:
                continue
            # paired bootstrap at each model's own CV-selected (balanced) threshold
            ts, tm = bundles[stack].thresholds["balanced"], bundles[s].thresholds["balanced"]
            rng = np.random.default_rng(cfg.seed)
            yv, pa, pb = y_test.to_numpy(), probas[stack], probas[s]
            from sklearn.metrics import f1_score
            diffs = []
            for _ in range(B):
                idx = rng.integers(0, len(yv), len(yv))
                diffs.append(f1_score(yv[idx], (pa[idx] >= ts).astype(int), zero_division=0)
                             - f1_score(yv[idx], (pb[idx] >= tm).astype(int), zero_division=0))
            f1_diff = {"diff": float(tbal.loc[stack, "f1"] - tbal.loc[s, "f1"]),
                       "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
                       "p_stack_better": float(np.mean(np.asarray(diffs) > 0))}
            ap_diff = paired_bootstrap_diff(y_test, pa, pb, average_precision_score, B, cfg.seed)
            stack_vs.append({
                "compared_to": s,
                "cv_f1@0.5_stack": cvs.loc[stack, "f1_mean"], "cv_f1@0.5_other": cvs.loc[s, "f1_mean"],
                "cv_f1@0.5_rel_pct": _pct(cvs.loc[stack, "f1_mean"], cvs.loc[s, "f1_mean"]),
                "cv_pr_auc_stack": cvs.loc[stack, "pr_auc_mean"], "cv_pr_auc_other": cvs.loc[s, "pr_auc_mean"],
                "oof_f1@tuned_stack": cvs.loc[stack, "oof_f1_at_balanced"], "oof_f1@tuned_other": cvs.loc[s, "oof_f1_at_balanced"],
                "test_f1@tuned_stack": tbal.loc[stack, "f1"], "test_f1@tuned_other": tbal.loc[s, "f1"],
                "test_f1@tuned_rel_pct": _pct(tbal.loc[stack, "f1"], tbal.loc[s, "f1"]),
                "test_f1@tuned_diff_ci95": [f1_diff["ci_low"], f1_diff["ci_high"]],
                "test_f1@tuned_p_stack_better": f1_diff["p_stack_better"],
                "test_f1@0.5_stack": t05.loc[stack, "f1"], "test_f1@0.5_other": t05.loc[s, "f1"],
                "test_pr_auc_diff": ap_diff["diff"], "test_pr_auc_diff_ci95": [ap_diff["ci_low"], ap_diff["ci_high"]],
            })
    write_json(stack_vs, metrics_dir / "stacking_vs_individual.json")

    # Fold-level paired comparison (same folds for every candidate)
    fold_cmp = {}
    if stack in cvs.index:
        piv = folds.pivot_table(index=["repeat", "fold"], columns="candidate", values=["f1", "pr_auc"])
        for s in singles:
            if s in cvs.index:
                fold_cmp[s] = {
                    "folds_stack_higher_f1@0.5": int((piv["f1"][stack] > piv["f1"][s]).sum()),
                    "folds_stack_higher_pr_auc": int((piv["pr_auc"][stack] > piv["pr_auc"][s]).sum()),
                    "n_folds": int(len(piv)),
                }
    write_json(fold_cmp, metrics_dir / "stacking_fold_wins.json")

    # Hypothesis checks: does SMOTE raise recall, does stacking beat its base learners?
    se = smote_effect.set_index("model")
    claims = {
        "claim_1_smote_recall": {
            "original_claim": "SMOTE improved minority-class recall over baseline models (~18%).",
            "measured_cv_recall@0.5_delta_pp": se["cv_recall@0.5_delta_pp"].round(2).to_dict(),
            "measured_cv_recall@0.5_relative_pct": se["cv_recall@0.5_rel_pct"].round(2).to_dict(),
            "measured_test_recall@0.5_delta_pp": se["test_recall@0.5_delta_pp"].round(2).to_dict(),
            "measured_cv_pr_auc_delta": se["cv_pr_auc_delta"].round(4).to_dict(),
        },
        "claim_2_stacking_f1": {
            "original_claim": "Stacking outperformed individual classifiers in F1 (~6-8%).",
            "comparisons": stack_vs,
            "fold_wins": fold_cmp,
        },
    }
    write_json(claims, metrics_dir / "claims_verification.json")

    # ---------------- 4. figures ----------------
    key = [c for c in [deployed, "stacking_nosmote_tuned", "xgboost_smote_tuned", "extratrees_smote_tuned",
                       "catboost_smote_tuned", "logistic_regression_nosmote_default"] if c in probas]
    labels = {c: short_label(c) for c in probas}
    curves = {c: (y_test.to_numpy(), probas[c]) for c in key}
    plot_roc_curves(curves, labels, figures_dir / "test_roc_curves.png", title="ROC curves - test set")
    plot_pr_curves(curves, labels, figures_dir / "test_pr_curves.png", title="Precision-recall curves - test set")
    plot_calibration_curves(curves, labels, figures_dir / "test_calibration_curves.png",
                            title="Calibration - test set (quantile bins)")
    plot_calibration_curves({f"{deployed}": (y_test.to_numpy(), p_dep),
                             "stacking_nosmote_raw_view": (y_test.to_numpy(), raw_dep)},
                            {deployed: f"{short_label(deployed)} - deployed ({dep.calibrator.method})",
                             "stacking_nosmote_raw_view": f"{short_label(deployed)} - uncalibrated"},
                            figures_dir / "test_calibration_raw_vs_calibrated.png",
                            title="Deployed model: raw vs calibrated probabilities (test)")
    cms = {c: (y_test.to_numpy(), (probas[c] >= bundles[c].thresholds["balanced"]).astype(int)) for c in key}
    plot_confusion_matrices(cms, {c: f"{labels[c]}\nthr={bundles[c].thresholds['balanced']:.2f}" for c in key},
                            figures_dir / "test_confusion_matrices.png")
    plot_confusion_matrices({m: (y_test.to_numpy(), (p_dep >= t).astype(int)) for m, t in dep.thresholds.items()},
                            {m: f"{m.replace('_', ' ')} (thr={t:.2f})" for m, t in dep.thresholds.items()},
                            figures_dir / "test_confusion_deployed_modes.png")
    bal = test_all[test_all.threshold_mode == "balanced"]
    plot_metric_bars(bal, "f1", labels=labels, path=figures_dir / "test_f1_comparison.png",
                     title="Test F1 at each model's CV-selected threshold")
    plot_metric_bars(bal, "recall", labels=labels, path=figures_dir / "test_recall_comparison.png",
                     title="Test recall at each model's CV-selected threshold")
    plot_metric_bars(cv_summary, "pr_auc_mean", err_col="pr_auc_std", labels=labels,
                     path=figures_dir / "cv_pr_auc_comparison.png", title="CV PR-AUC (mean +/- std over 15 folds)")
    plot_metric_bars(cv_summary, "f1_mean", err_col="f1_std", labels=labels,
                     path=figures_dir / "cv_f1_comparison.png", title="CV F1 at threshold 0.5 (mean +/- std)")
    plot_metric_bars(cv_summary, "recall_mean", err_col="recall_std", labels=labels,
                     path=figures_dir / "cv_recall_comparison.png", title="CV recall at threshold 0.5 (mean +/- std)")
    order = cv_summary.sort_values("pr_auc_mean")["candidate"]
    plot_cv_boxplots(folds, "pr_auc", order, labels, figures_dir / "cv_boxplot_pr_auc.png")
    plot_cv_boxplots(folds, "f1", order, labels, figures_dir / "cv_boxplot_f1.png")
    thr_table = pd.read_csv(metrics_dir / f"threshold_table_{deployed}.csv")
    plot_threshold_tradeoff(thr_table, dep.thresholds, figures_dir / "threshold_tradeoff_deployed.png",
                            title=f"Threshold trade-off - {short_label(deployed)} (training OOF)")
    plot_probability_histogram(y_test, p_dep, thr_bal, figures_dir / "test_probability_histogram.png")

    # ---------------- 5. error analysis ----------------
    feats = cfg.features
    err_test = ea.error_table(X_test, y_test, p_dep, thr_bal, zero_cols)
    err_test.to_csv(metrics_dir / "errors_test.csv", index_label="row_id")
    ea.error_profile(err_test, feats).to_csv(metrics_dir / "error_profile_test.csv")
    ea.borderline_cases(err_test).to_csv(metrics_dir / "borderline_cases_test.csv", index_label="row_id")
    ea.plot_error_features(err_test, feats, figures_dir / "errors_test_features.png",
                           title="Test set: features by outcome type")
    ea.plot_error_probability(err_test, thr_bal, figures_dir / "errors_test_probability.png")
    # OOF (training) errors: 3x more patients -> more stable patterns. Uses repeat-0 OOF probabilities
    # mapped through the deployed calibrator (calibration nested effects are negligible here).
    oof_dep = oof[(oof.candidate == deployed) & (oof.repeat == 0)].set_index("row_id").loc[X_train.index]
    p_oof = dep.calibrator.transform(oof_dep["proba"].to_numpy())
    err_oof = ea.error_table(X_train, y_train, p_oof, thr_bal, zero_cols)
    err_oof.to_csv(metrics_dir / "errors_train_oof.csv", index_label="row_id")
    ea.error_profile(err_oof, feats).to_csv(metrics_dir / "error_profile_train_oof.csv")
    ea.missed_vs_caught(err_oof, feats).to_csv(metrics_dir / "fn_vs_tp_train_oof.csv", index=False)
    ea.plot_error_features(err_oof, feats, figures_dir / "errors_oof_features.png",
                           title="Training OOF: features by outcome type")
    error_summary = {"test": ea.summarize_errors(err_test, thr_bal), "train_oof": ea.summarize_errors(err_oof, thr_bal)}
    write_json(error_summary, metrics_dir / "error_analysis_summary.json")

    # ---------------- 6. explainability ----------------
    shap_summary = {}
    if run_shap:
        Xtr_imp = impute_like_training(X_train, dep, zero_cols)
        bg = background_sample(Xtr_imp, y_train, cfg["explainability"]["background_size"], cfg.seed)
        if not (models_dir / "shap_background.csv").exists():
            bg.to_csv(models_dir / "shap_background.csv", index=False)
        explainer = ShapExplainer(dep, bg)
        Xte_imp = impute_like_training(X_test, dep, zero_cols).iloc[: cfg["explainability"]["max_global_samples"]]
        exp = explainer.explain(Xte_imp)
        pd.DataFrame(exp.values, columns=dep.features, index=Xte_imp.index).to_csv(
            metrics_dir / "shap_values_test.csv", index_label="row_id")
        gi = global_importance(exp)
        gi.to_csv(metrics_dir / "shap_global_importance.csv", index=False)
        save_global_plots(exp, figures_dir)
        # additivity check: base + sum(SHAP) must equal the model output
        recon = exp.base_values + exp.values.sum(axis=1)
        additivity_err = float(np.max(np.abs(recon - dep.predict_proba(Xte_imp))))
        examples = {}
        pos = {rid: i for i, rid in enumerate(Xte_imp.index)}
        for et in ["TP", "FN", "FP"]:
            ids = [r for r in err_test.index[err_test.error_type == et] if r in pos]
            if len(ids):
                rid = err_test.loc[ids, "proba"].sort_values().index[len(ids) // 2]
                i = pos[rid]
                save_waterfall(exp[i], figures_dir / f"shap_waterfall_{et}.png",
                               title=f"Example {et} (test row {rid}): p={p_dep[X_test.index.get_loc(rid)]:.2f}")
                examples[et] = {"row_id": int(rid), "probability": float(p_dep[X_test.index.get_loc(rid)]),
                                "contributions": contributions(exp[i])}
        shap_summary = {"expected_value": float(np.mean(exp.base_values)), "global_importance": gi.to_dict("records"),
                        "additivity_max_abs_error": additivity_err, "examples": examples,
                        "background_size": len(bg), "n_explained": len(Xte_imp)}
        write_json(shap_summary, metrics_dir / "shap_summary.json")
        # FN vs TP mean SHAP on test (which contributions pushed missed cases down?)
        sv = pd.DataFrame(exp.values, columns=dep.features, index=Xte_imp.index)
        sv["error_type"] = err_test.loc[sv.index, "error_type"]
        sv.groupby("error_type").mean().to_csv(metrics_dir / "shap_mean_by_error_type_test.csv")

    perm = permutation_importance_ap(dep, X_test, y_test, n_repeats=20, seed=cfg.seed)
    perm.to_csv(metrics_dir / "permutation_importance_test.csv", index=False)
    plot_importance_bars(perm, "pr_auc_drop_mean", "pr_auc_drop_std",
                         "Permutation importance (drop in test PR-AUC)", figures_dir / "permutation_importance.png")
    native = native_importances(dep)
    native.to_csv(metrics_dir / "native_importances_base_learners.csv")

    # ---------------- 7. stability ----------------
    stab_path = metrics_dir / "stability_seeds.csv"
    stability = {}
    if stab_path.exists():
        st = pd.read_csv(stab_path)
        agg = st.groupby("candidate")[["pr_auc", "roc_auc", "f1", "recall", "brier"]].agg(["mean", "std", "min", "max"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        agg.to_csv(metrics_dir / "stability_summary.csv")
        stability = agg.round(4).reset_index().to_dict("records")
        # does the ranking stacking vs best single base flip across seeds?
        piv = st.pivot(index="seed", columns="candidate", values="pr_auc")
        if stack in piv:
            base_cols = [c for c in singles if c in piv]
            stability_rank = {"seeds_where_stack_beats_every_single_pr_auc":
                              int((piv[stack] > piv[base_cols].max(axis=1)).sum()),
                              "n_seeds": int(len(piv))}
            write_json(stability_rank, metrics_dir / "stability_ranking.json")

    # ---------------- 8. metadata, log, tracking ----------------
    meta_path = models_dir / "model_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["test_metrics"] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_test": int(len(y_test)),
        "balanced": dep_modes["balanced"],
        "high_sensitivity": dep_modes["high_sensitivity"],
        "high_precision": dep_modes["high_precision"],
        "default_0.5": dep_modes["default_0.5"],
        "bootstrap_95ci_at_balanced": deployed_test["bootstrap_95ci_at_balanced"],
    }
    write_json(meta, meta_path)
    with open(metrics_dir / "test_evaluation_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"evaluated_at": meta["test_metrics"]["evaluated_at"], "deployed_model": deployed,
                             "model_sha256": sha256_of(models_dir / "model.joblib")}) + "\n")
    tracker = ExperimentTracker(cfg)
    with tracker.run(f"test-evaluation-{deployed}", tags={"stage": "final_test"}):
        tracker.log_metrics(dep_modes["balanced"], prefix="test_balanced_")
        tracker.log_metrics(dep_modes["high_sensitivity"], prefix="test_high_sens_")
        tracker.log_params({"deployed_model": deployed, "thresholds": dep.thresholds})
        tracker.log_artifact(metrics_dir / "final_comparison_table.csv")

    write_results_summary(cfg, deployed, final, smote_table, smote_effect, stack_vs, deployed_test, stability,
                          shap_summary, error_summary)
    logger.info("Evaluation complete. Deployed %s: test F1=%.3f recall=%.3f AUC=%.3f",
                deployed, dep_modes["balanced"]["f1"], dep_modes["balanced"]["recall"], dep_modes["balanced"]["roc_auc"])
    return {"deployed": deployed_test, "final": final}


def _md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_results_summary(cfg, deployed, final, smote_table, smote_effect, stack_vs, deployed_test, stability,
                          shap_summary, error_summary) -> None:
    """Auto-generated tables (every number comes straight from the evaluation run)."""
    lines = [
        "# Results summary (auto-generated by `scripts/evaluate.py`)",
        "",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Deployed model: `{deployed}`.",
        "Do not edit by hand; re-run `python scripts/evaluate.py`.",
        "",
        "## Final test-set comparison (each model at its CV-selected balanced threshold)",
        "",
        _md_table(final.drop(columns=["display_name"]).assign(selected=final.selected.map({True: "**selected**", False: ""}))),
        "",
        "## Deployed model by operating mode (test set)",
        "",
        _md_table(pd.DataFrame([{"mode": m, **{k: v[k] for k in ["threshold", "precision", "recall", "f1", "specificity", "tp", "fp", "fn", "tn"]}}
                                for m, v in deployed_test["metrics_by_threshold_mode"].items()])),
        "",
        "Bootstrap 95% CIs (balanced threshold): " + ", ".join(
            f"{k} {v['point']:.3f} [{v['low']:.3f}, {v['high']:.3f}]" for k, v in deployed_test["bootstrap_95ci_at_balanced"].items()),
        "",
        "## SMOTE vs no SMOTE",
        "",
        _md_table(smote_table[["model", "smote", "cv_precision@0.5", "cv_recall@0.5", "cv_f1@0.5", "cv_roc_auc", "cv_pr_auc",
                               "test_precision@0.5", "test_recall@0.5", "test_f1@0.5", "test_roc_auc", "test_pr_auc"]]),
        "",
        "### SMOTE effect (SMOTE minus no-SMOTE)",
        "",
        _md_table(smote_effect),
        "",
        "## Stacking vs individual learners",
        "",
        _md_table(pd.DataFrame(stack_vs).drop(columns=["test_f1@tuned_diff_ci95", "test_pr_auc_diff_ci95"])) if stack_vs else "n/a",
        "",
        "## Calibration (deployed)",
        "",
        f"Method: {deployed_test['calibration']['method']}; test Brier raw {deployed_test['calibration']['brier_raw']:.4f} "
        f"vs calibrated {deployed_test['calibration']['brier_calibrated']:.4f}",
        "",
    ]
    if shap_summary:
        lines += ["## SHAP global importance (mean |SHAP|, probability units)", "",
                  _md_table(pd.DataFrame(shap_summary["global_importance"]), ".4f"), ""]
    if stability:
        lines += ["## Seed stability (5 seeds x 5-fold CV)", "", _md_table(pd.DataFrame(stability), ".4f"), ""]
    lines += ["## Error analysis", "", "```json", json.dumps(error_summary, indent=2, default=str), "```", ""]
    (cfg.path("paths", "reports_dir") / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
