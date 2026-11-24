"""End-to-end training: validation -> split -> CV experiments -> tuning -> calibration ->
threshold selection -> final fit -> artifacts.

Nothing in this module reads ``data/processed/test.csv`` beyond writing it. All selection
decisions (hyperparameters, calibration, thresholds, deployed model) are made from training
data via cross-validation. The test set is scored once, later, by ``src.evaluation.evaluate``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import __version__
from src.config import Config, set_global_seed
from src.data.eda import run_eda
from src.data.loader import load_dataset, save_split, split_data
from src.data.preprocessing import ZeroToNaN, training_medians
from src.data.validation import validate_dataframe
from src.evaluation.metrics import (
    compute_metrics,
    select_thresholds,
    summarize_folds,
    threshold_table,
)
from src.models.bundle import ModelBundle
from src.models.calibration import ProbabilityCalibrator, nested_calibrated_oof, select_calibration
from src.models.registry import CandidateSpec
from src.models.stacking_model import BASE_LEARNERS
from src.training.cross_validation import CVResult, cross_validate_model
from src.training.tracking import ExperimentTracker
from src.training.tune import base_oof_meta_features, tune_base_learner, tune_meta_learner

logger = logging.getLogger(__name__)

SINGLE_FAMILIES = ["logistic_regression", "random_forest", "xgboost", "extratrees", "catboost"]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def library_versions() -> dict[str, str]:
    import catboost
    import imblearn
    import shap
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "imbalanced-learn": imblearn.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
        "shap": shap.__version__,
    }


def quick_overrides() -> dict:
    """Small settings for smoke tests / CI (never used for reported results)."""
    return {
        "cv": {"n_splits": 3, "n_repeats": 1, "stacking_inner_splits": 3},
        "tuning": {"n_splits": 3, "n_trials": {"xgboost": 2, "extratrees": 2, "catboost": 2}, "meta_n_trials": 3},
        "models": {
            "random_forest": {"n_estimators": 50},
            "xgboost": {"n_estimators": 50},
            "extratrees": {"n_estimators": 50},
            "catboost": {"iterations": 50},
        },
        "stability": {"enabled": False},
        "mlflow": {"enabled": False},
    }


# --------------------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------------------
def tune_all(X: pd.DataFrame, y: pd.Series, cfg: Config, metrics_dir: Path) -> dict:
    """Optuna-tune each base learner (with SMOTE) and the meta learner."""
    base_params: dict[str, dict] = {}
    summary: dict[str, Any] = {"metric": cfg["tuning"]["metric"], "base": {}}
    for name in BASE_LEARNERS:
        t0 = time.perf_counter()
        best, trials = tune_base_learner(name, X, y, cfg, n_trials=cfg["tuning"]["n_trials"][name])
        trials.to_csv(metrics_dir / f"tuning_trials_{name}.csv", index=False)
        base_params[name] = best
        summary["base"][name] = {
            "best_params": best,
            "best_cv_score": float(trials["value"].max()),
            "n_trials": len(trials),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        logger.info("Tuned %s in %.0fs", name, time.perf_counter() - t0)

    Z = base_oof_meta_features(base_params, X, y, cfg)
    meta_params, meta_trials = tune_meta_learner(Z, y, cfg, cfg["tuning"]["meta_n_trials"])
    meta_trials.to_csv(metrics_dir / "tuning_trials_meta.csv", index=False)
    summary["meta"] = {"best_params": meta_params, "best_cv_score": float(meta_trials["value"].max())}
    return {"base_params": base_params, "meta_params": meta_params, "summary": summary, "meta_features": Z}


def stacking_analysis(Z: pd.DataFrame, y: pd.Series, meta_params: dict, cfg: Config) -> dict:
    """How much does each base learner contribute to the meta learner?

    Uses the out-of-fold meta-feature matrix (what the meta learner is trained on):
    correlations (diversity), standardized meta coefficients, and drop-one ablation.
    """
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

    cv = StratifiedKFold(cfg["tuning"]["n_splits"], shuffle=True, random_state=cfg["tuning"]["cv_seed"])

    def meta_cv(cols):
        lr = LogisticRegression(max_iter=5000, **meta_params)
        p = cross_val_predict(lr, Z[cols], y, cv=cv, method="predict_proba")[:, 1]
        return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
                "f1@0.5": f1_score(y, (p >= 0.5).astype(int))}

    full = meta_cv(list(Z.columns))
    lr = LogisticRegression(max_iter=5000, **meta_params).fit(Z, y)
    coefs = dict(zip(Z.columns, lr.coef_[0]))
    std_coefs = {k: float(v * Z[k].std()) for k, v in coefs.items()}
    out = {
        "note": "Computed on out-of-fold base-learner probabilities from the training set only.",
        "oof_correlation": Z.corr().round(4).to_dict(),
        "meta_coefficients": {k: float(v) for k, v in coefs.items()},
        "meta_intercept": float(lr.intercept_[0]),
        "meta_standardized_coefficients": std_coefs,
        "all_bases": full,
        "drop_one": {},
        "single_base_raw": {},
    }
    for name in Z.columns:
        rest = [c for c in Z.columns if c != name]
        res = meta_cv(rest)
        out["drop_one"][name] = {**res, "delta_pr_auc": res["pr_auc"] - full["pr_auc"]}
        out["single_base_raw"][name] = {
            "pr_auc": average_precision_score(y, Z[name]),
            "roc_auc": roc_auc_score(y, Z[name]),
            "f1@0.5": f1_score(y, (Z[name] >= 0.5).astype(int)),
        }
    return out


def calibrate_and_threshold(cv: CVResult, cfg: Config) -> dict:
    """Choose calibration (nested CV Brier) and the three thresholds from OOF predictions."""
    oof = cv.oof
    proba, y, groups = oof["proba"].to_numpy(), oof["y_true"].to_numpy(), oof["row_id"].to_numpy()
    method, brier_scores = select_calibration(
        proba, y, groups, cfg["calibration"]["methods"], cfg["calibration"]["min_brier_improvement"]
    )
    calibrator = ProbabilityCalibrator(method).fit(proba, y)
    # Thresholds are chosen on *nested* calibrated OOF probabilities, so the calibrator that
    # produced each probability never saw that patient.
    p_eval = proba if method == "none" else nested_calibrated_oof(proba, y, groups, method)
    tcfg = cfg["threshold"]
    table = threshold_table(y, p_eval, tcfg["grid_start"], tcfg["grid_stop"], tcfg["grid_step"])
    chosen = select_thresholds(table, tcfg["high_sensitivity_min_recall"], tcfg["high_precision_min_precision"])
    oof_at = {mode: compute_metrics(y, p_eval, v["threshold"]) for mode, v in chosen.items()}
    oof_at["default_0.5"] = compute_metrics(y, p_eval, 0.5)
    return {
        "calibration_method": method,
        "calibration_nested_brier": brier_scores,
        "calibrator": calibrator,
        "thresholds": {m: v["threshold"] for m, v in chosen.items()},
        "threshold_rules": chosen,
        "threshold_table": table,
        "oof_metrics_at_thresholds": oof_at,
    }


def run_stability(specs: list[CandidateSpec], X, y, cfg: Config) -> pd.DataFrame:
    """Seed sensitivity: re-run 5-fold CV with different CV-split *and* model seeds."""
    rows = []
    for seed in cfg["stability"]["seeds"]:
        for spec in specs:
            res = cross_validate_model(spec.build(cfg, seed=seed), X, y, n_splits=cfg["cv"]["n_splits"],
                                       n_repeats=1, seed=seed, n_jobs=cfg["cv"]["n_jobs"])
            m = res.fold_metrics.mean(numeric_only=True)
            rows.append({"seed": seed, "candidate": spec.name,
                         **{k: m[k] for k in ["pr_auc", "roc_auc", "f1", "recall", "precision", "brier"]}})
            logger.info("stability seed=%s %s AP=%.4f", seed, spec.name, m["pr_auc"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------------------
def run_training(cfg: Config, reuse_tuning: bool = False, skip_eda: bool = False) -> dict:
    t_start = time.perf_counter()
    set_global_seed(cfg.seed)
    metrics_dir, figures_dir, models_dir = cfg.path("paths", "metrics_dir"), cfg.path("paths", "figures_dir"), cfg.path("paths", "models_dir")
    for d in (metrics_dir, figures_dir, models_dir / "candidates"):
        d.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(cfg)

    # ---------------- 1. ingestion + validation ----------------
    df = load_dataset(cfg)
    report = validate_dataframe(df, cfg.features, cfg.target, cfg["data"]["zero_as_missing"])
    write_json(report.to_dict(), metrics_dir / "data_validation.json")
    if not report.is_valid:
        raise ValueError(f"Data validation failed: {report.errors}")
    logger.info("Validated dataset: %s rows, %s duplicates", report.n_rows, report.n_duplicate_rows)
    if not skip_eda:
        run_eda(df, cfg["data"]["zero_as_missing"], figures_dir, metrics_dir)

    # ---------------- 2. split (before any resampling) ----------------
    X_train, X_test, y_train, y_test = split_data(df, cfg)
    save_split(X_train, X_test, y_train, y_test, cfg)
    split_info = {
        "n_train": len(X_train), "n_test": len(X_test),
        "train_prevalence": float(y_train.mean()), "test_prevalence": float(y_test.mean()),
        "seed": cfg.seed, "test_size": cfg["split"]["test_size"],
    }
    write_json(split_info, metrics_dir / "split_info.json")

    # ---------------- 3. hyperparameter optimisation (training data only) ----------------
    tuning_path = metrics_dir / "tuning_results.json"
    if cfg["tuning"]["enabled"]:
        if reuse_tuning and tuning_path.exists():
            tuned = json.loads(tuning_path.read_text())
            base_params, meta_params = tuned["base_params"], tuned["meta_params"]
            Z = base_oof_meta_features(base_params, X_train, y_train, cfg)
            logger.info("Reusing tuned hyperparameters from %s", tuning_path)
        else:
            tuned_all = tune_all(X_train, y_train, cfg, metrics_dir)
            base_params, meta_params, Z = tuned_all["base_params"], tuned_all["meta_params"], tuned_all["meta_features"]
            write_json({"base_params": base_params, "meta_params": meta_params, **tuned_all["summary"]}, tuning_path)
    else:
        base_params = {n: {} for n in BASE_LEARNERS}
        meta_params = {}
        Z = base_oof_meta_features(base_params, X_train, y_train, cfg)
    write_json(stacking_analysis(Z, y_train, meta_params, cfg), metrics_dir / "stacking_analysis.json")

    # ---------------- 4. candidate definitions ----------------
    specs: list[CandidateSpec] = []
    for fam in SINGLE_FAMILIES + ["stacking"]:
        for smote in (False, True):
            specs.append(CandidateSpec(fam, smote=smote, tuned=False))
    if cfg["tuning"]["enabled"]:
        for fam in BASE_LEARNERS:
            for smote in (False, True):
                specs.append(CandidateSpec(fam, smote=smote, tuned=True, params=base_params[fam]))
        for smote in (False, True):
            specs.append(CandidateSpec("stacking", smote=smote, tuned=True, base_params=base_params,
                                       meta_params=meta_params))
        specs.append(CandidateSpec("soft_voting", smote=True, tuned=True, base_params=base_params))

    # ---------------- 5. repeated stratified CV for every candidate ----------------
    fold_frames, oof_frames, results = [], [], {}
    with tracker.run(f"training-{datetime.now():%Y%m%d-%H%M%S}"):
        tracker.log_params({"seed": cfg.seed, "cv": cfg["cv"], "preprocessing": cfg["preprocessing"], "smote": cfg["smote"]})
        for spec in specs:
            cv_res = cross_validate_model(
                spec.build(cfg), X_train, y_train, n_splits=cfg["cv"]["n_splits"],
                n_repeats=cfg["cv"]["n_repeats"], seed=cfg.seed, n_jobs=cfg["cv"]["n_jobs"],
            )
            post = calibrate_and_threshold(cv_res, cfg)
            results[spec.name] = {"spec": spec, "cv": cv_res, "post": post}
            fold_frames.append(cv_res.fold_metrics.assign(candidate=spec.name))
            oof_frames.append(cv_res.oof.assign(candidate=spec.name))
            mean = cv_res.fold_metrics.mean(numeric_only=True)
            logger.info("CV %-36s AP=%.4f AUC=%.4f F1@.5=%.4f Rec@.5=%.4f (%.0fs)", spec.name, mean["pr_auc"],
                        mean["roc_auc"], mean["f1"], mean["recall"], cv_res.fit_seconds)
            with tracker.run(spec.name, nested=True, tags={"smote": str(spec.smote), "family": spec.family}):
                tracker.log_params({"spec": spec.to_dict(), "calibration": post["calibration_method"],
                                    "thresholds": post["thresholds"]})
                tracker.log_metrics(mean.drop(["repeat", "fold"]).to_dict(), prefix="cv_")
                tracker.log_metrics(post["oof_metrics_at_thresholds"]["balanced"], prefix="oof_balanced_")

        folds = pd.concat(fold_frames, ignore_index=True)
        folds.to_csv(metrics_dir / "cv_fold_metrics.csv", index=False)
        pd.concat(oof_frames, ignore_index=True).to_csv(metrics_dir / "cv_oof_predictions.csv", index=False)
        summary = summarize_folds(folds)
        meta_rows = []
        for name, r in results.items():
            s, post = r["spec"], r["post"]
            bal = post["oof_metrics_at_thresholds"]["balanced"]
            meta_rows.append({
                "candidate": name, "display_name": s.display_name, "family": s.family, "smote": s.smote,
                "tuned": s.tuned, "calibration": post["calibration_method"],
                "threshold_balanced": post["thresholds"]["balanced"],
                "threshold_high_sensitivity": post["thresholds"]["high_sensitivity"],
                "threshold_high_precision": post["thresholds"]["high_precision"],
                "oof_f1_at_balanced": bal["f1"], "oof_recall_at_balanced": bal["recall"],
                "oof_precision_at_balanced": bal["precision"],
                "fit_seconds": r["cv"].fit_seconds,
            })
        summary = summary.merge(pd.DataFrame(meta_rows), on="candidate")
        summary.to_csv(metrics_dir / "cv_summary.csv", index=False)
        pd.DataFrame([{"candidate": n, **{f"brier_{k}": v for k, v in r["post"]["calibration_nested_brier"].items()},
                       "chosen": r["post"]["calibration_method"]} for n, r in results.items()]
                     ).to_csv(metrics_dir / "calibration_selection.csv", index=False)

        # ---------------- 6. model selection (a-priori rule, training data only) ----------------
        sel = cfg["selection"]
        metric_col = {"average_precision": "pr_auc"}.get(sel["metric"], sel["metric"]) + "_mean"
        best_cv = summary.sort_values(metric_col, ascending=False).iloc[0]["candidate"]
        deployed = sel["fixed_model"] if sel["mode"] == "fixed" else best_cv
        if deployed not in results:
            raise KeyError(f"Selected model '{deployed}' was not trained (is tuning disabled?)")
        selection = {
            "mode": sel["mode"], "deployed_model": deployed, "best_by_cv": best_cv,
            "selection_metric": metric_col,
            "deployed_cv_score": float(summary.set_index("candidate").loc[deployed, metric_col]),
            "best_cv_score": float(summary.set_index("candidate").loc[best_cv, metric_col]),
            "note": ("Deployed architecture fixed a priori by the project brief; "
                     "best_by_cv reported for transparency.") if sel["mode"] == "fixed" else "Selected by CV.",
        }
        write_json(selection, metrics_dir / "model_selection.json")
        for name, r in results.items():
            r["post"]["threshold_table"].to_csv(metrics_dir / f"threshold_table_{name}.csv", index=False)

        # ---------------- 7. final fit of every candidate on the full training set ----------------
        medians = training_medians(X_train, cfg["data"]["zero_as_missing"])
        cleaned = ZeroToNaN(cfg["data"]["zero_as_missing"]).fit(X_train).transform(X_train)
        ranges = {c: (float(cleaned[c].min()), float(cleaned[c].max())) for c in cfg.features}
        config_hash = hashlib.sha256(json.dumps(dict(cfg), sort_keys=True, default=str).encode()).hexdigest()[:8]
        trained_at = datetime.now(timezone.utc)
        deployed_bundle = None
        for name, r in results.items():
            spec, post = r["spec"], r["post"]
            model = spec.build(cfg).fit(X_train, y_train)
            mean = r["cv"].fold_metrics.mean(numeric_only=True)
            std = r["cv"].fold_metrics.std(numeric_only=True)
            metadata = {
                "model_name": name,
                "display_name": spec.display_name,
                "model_version": f"{name}-{trained_at:%Y%m%d}-{config_hash}",
                "package_version": __version__,
                "dataset_version": cfg["data"]["version"],
                "dataset_sha256": cfg["data"]["sha256"],
                "training_date": trained_at.isoformat(),
                "features": cfg.features,
                "target": cfg.target,
                "random_seed": cfg.seed,
                "n_train": len(X_train),
                "train_prevalence": float(y_train.mean()),
                "preprocessing": {
                    "version": cfg["preprocessing"]["version"],
                    "zero_as_missing": cfg["data"]["zero_as_missing"],
                    "imputation": "median (fitted on training data inside each pipeline)",
                    "scaling": "StandardScaler for logistic regression only",
                    "engineered_features": cfg["preprocessing"]["engineered_features"],
                },
                "smote": {"used": spec.smote, **cfg["smote"],
                          "applied_to": "training folds only (imblearn pipeline, skipped at predict time)"},
                "hyperparameters": spec.to_dict(),
                "calibration": {"method": post["calibration_method"],
                                "nested_cv_brier": post["calibration_nested_brier"]},
                "thresholds": post["thresholds"],
                "threshold_rules": post["threshold_rules"],
                "default_threshold_mode": cfg["threshold"]["default_mode"],
                "cv_metrics": {
                    "scheme": f"{cfg['cv']['n_repeats']}x repeated stratified {cfg['cv']['n_splits']}-fold on training set, threshold 0.5",
                    "mean": {k: float(v) for k, v in mean.drop(["repeat", "fold"]).items()},
                    "std": {k: float(v) for k, v in std.drop(["repeat", "fold"]).items()},
                },
                "oof_metrics_at_selected_thresholds": post["oof_metrics_at_thresholds"],
                "selection": selection,
                "library_versions": library_versions(),
            }
            if spec.family == "stacking":
                metadata["meta_learner"] = {
                    "coefficients": dict(zip(BASE_LEARNERS, map(float, model.final_estimator_.coef_[0]))),
                    "intercept": float(model.final_estimator_.intercept_[0]),
                }
            bundle = ModelBundle(
                name=name, model=model, calibrator=post["calibrator"], thresholds=post["thresholds"],
                default_mode=cfg["threshold"]["default_mode"], features=cfg.features,
                imputation_medians=medians, training_ranges=ranges, metadata=metadata,
            )
            bundle.save(models_dir / "candidates" / f"{name}.joblib")
            if name == deployed:
                deployed_bundle = bundle

        deployed_bundle.save(models_dir / "model.joblib")
        # SHAP background: stratified sample of imputed training patients (used by API + evaluation)
        from src.explainability.shap_analysis import background_sample, impute_like_training

        background_sample(
            impute_like_training(X_train, deployed_bundle, cfg["data"]["zero_as_missing"]), y_train,
            cfg["explainability"]["background_size"], cfg.seed,
        ).to_csv(models_dir / "shap_background.csv", index=False)
        write_json(deployed_bundle.metadata, models_dir / "model_metadata.json")
        tracker.log_artifact(models_dir / "model_metadata.json")
        tracker.log_metrics(deployed_bundle.metadata["cv_metrics"]["mean"], prefix="deployed_cv_")

    # ---------------- 8. seed sensitivity ----------------
    if cfg["stability"]["enabled"]:
        stab_specs = [results[n]["spec"] for n in (
            "logistic_regression_nosmote_default", "xgboost_smote_tuned", "extratrees_smote_tuned",
            "catboost_smote_tuned", "stacking_smote_tuned", "stacking_nosmote_tuned") if n in results]
        stab = run_stability(stab_specs, X_train, y_train, cfg)
        stab.to_csv(metrics_dir / "stability_seeds.csv", index=False)

    elapsed = time.perf_counter() - t_start
    write_json({"seconds": round(elapsed, 1), "finished": datetime.now(timezone.utc).isoformat(),
                "deployed_model": deployed}, metrics_dir / "training_run.json")
    logger.info("Training finished in %.1f min. Deployed: %s", elapsed / 60, deployed)
    return {"selection": selection, "summary": summary}
