# Smart Health: Diabetes Prediction using a Stacking Ensemble — Final Report

*Version 1.0.0 · model `stacking_smote_tuned-20261006-17ecf561` · dataset `pima-uci-768-v1`*

> This is an ML research/educational decision-support prototype. It is **not clinically validated**
> and is not a medical diagnostic device.

Every number in this report was produced by `scripts/train.py` / `scripts/evaluate.py` and can be
traced to a file in `reports/metrics/`. The auto-generated tables are in
[`results_summary.md`](results_summary.md).

---

## 1. Summary

| Question | Answer (measured) |
|---|---|
| Does SMOTE improve minority recall? | **At the default 0.5 threshold, yes** (CV recall +3.4 to +26.9 pp across 10 model pairs). **It does not improve ranking quality** (CV PR-AUC change −0.013 to +0.003) and it worsens single-model Brier scores. The recall gain is mostly a probability shift that threshold tuning reproduces. |
| Does SMOTE improve F1? | At 0.5, yes in CV (+1.5 to +10.1 pp). At each model's tuned threshold, test F1 changes are mixed (−2.5 to +3.1 pp). |
| Does stacking beat the individual models? | **Not demonstrably.** Its CV PR-AUC (0.743) is within noise of tuned ExtraTrees (0.747) and below ExtraTrees without SMOTE (0.754, the best CV candidate). On the test set, F1 at tuned thresholds is +0.9 % to +4.7 % relative, but **every paired-bootstrap 95 % CI includes 0**. |
| Which base model contributes most? | ExtraTrees and CatBoost (standardized meta coefficients 0.61 / 0.64). XGBoost adds nothing: dropping it changes CV PR-AUC by +0.0005. |
| Does the ensemble improve calibration? | Relative to SMOTE-trained single models, yes in CV (Brier 0.154 vs 0.162–0.171; ECE 0.081 vs 0.106–0.140), because the meta-learner is fitted on real-prevalence OOF data. It is no better than plain logistic regression (0.152). No post-hoc calibration was adopted. |
| Best sensitivity/F1 threshold | **0.32** (max OOF F1) and **0.25** for high sensitivity (OOF recall ≥ 0.85). Both are well below 0.5. |
| Dominant features | Glucose (mean \|SHAP\| 0.147), then BMI (0.075), Age (0.047) and DiabetesPedigreeFunction (0.041). |
| Who is misclassified? | Missed diabetics (false negatives) mostly have glucose below 126 mg/dL (80 % vs 21 % of caught diabetics). They are also younger and leaner. False positives mostly have glucose ≥ 126. |
| Seed sensitivity | Stacking PR-AUC across 5 seeds: 0.733 ± 0.011. Stacking beat every single base learner on PR-AUC in **0 of 5** seeds. |
| CV-fold stability | The fold-to-fold spread (PR-AUC 0.642–0.857, std 0.060) is about 10× larger than the differences between top models. |

**Deployed model (test set, n = 154, balanced threshold 0.32):** accuracy 0.747, precision 0.603,
**recall/sensitivity 0.815**, specificity 0.710, **F1 0.693**, ROC-AUC 0.821, PR-AUC 0.709, Brier 0.166.
Bootstrap 95 % CIs: F1 [0.598, 0.777], recall [0.714, 0.917], ROC-AUC [0.748, 0.880].

---

## 2. Data

768 patients, 8 features, 34.9 % positive. There were no duplicates and no explicit NaNs. Physiologically
impossible zeros were found in Glucose (5), BloodPressure (35), SkinThickness (227), Insulin (374) and
BMI (11), and were treated as missing. Zeros in Pregnancies (111) are genuine. All decisions are documented in
[`data/README.md`](../data/README.md).

Descriptive differences between classes (Mann-Whitney U, rank-biserial r; association only, not causation):
Glucose r = 0.59 (median 140 vs 107 mg/dL), Insulin 0.46, BMI 0.37, Age 0.37 (median 36 vs 27 years),
SkinThickness 0.33, Pregnancies 0.24, BloodPressure 0.22, DiabetesPedigreeFunction 0.21. All p < 1e-5.

## 3. Methodology

### 3.1 Protocol (what touched which data)

```
768 patients ─ stratified 80/20 split (seed 42) ─┬─ train 614 (34.85 % positive)
                                                 │    ├─ Optuna tuning (5-fold CV, seed 7)
                                                 │    ├─ 3×5 repeated stratified CV of 21 candidates (seed 42)
                                                 │    ├─ calibration choice (nested CV Brier on OOF)
                                                 │    ├─ threshold choice (OOF predictions)
                                                 │    ├─ seed-stability study (5 seeds × 5-fold)
                                                 │    └─ final refit on all 614
                                                 └─ test 154 (35.06 % positive) ── scored once
```

### 3.2 Leakage controls

- **Imputation inside each pipeline.** Each estimator is an `imblearn` pipeline:
  `ZeroToNaN → SimpleImputer(median) → [SMOTE] → model`. Medians and SMOTE neighbours are recomputed on
  every training fold, every stacking inner fold and the final fit.
- **SMOTE never sees validation rows.** It runs only in `fit` (imblearn skips samplers at predict time).
  A unit test asserts that the classifier trains on 2 × 400 = 800 rows and predicts exactly `len(X_test)` rows.
- **Out-of-fold meta-features.** `StackingClassifier(cv=StratifiedKFold(5), stack_method="predict_proba")`
  trains the meta-learner only on out-of-fold probabilities of **real** patients. SMOTE sits *inside* each
  base pipeline, not in front of the stack, so no synthetic sample interpolated from an inner-training
  patient can appear in an inner validation fold.
- **Selections use training data only.** Hyperparameters, calibration method, thresholds and the deployed
  architecture were all chosen on training data.
- **Single scoring of the test set.** The test set was scored once with the final code (logged in
  `reports/metrics/test_evaluation_log.jsonl`). *Disclosure:* while debugging the evaluation code, one
  smoke-test run with toy settings (50-tree models, 3-fold CV) was scored against the same test split. No
  setting, threshold or model choice was changed afterwards; all selection rules were fixed in
  `configs/config.yaml` beforehand.

### 3.3 Models

| Family | Variants evaluated |
|---|---|
| Logistic Regression (StandardScaler) | default, ± SMOTE |
| Random Forest | default, ± SMOTE |
| XGBoost, ExtraTrees, CatBoost | default and Optuna-tuned, ± SMOTE |
| Stacking (XGB + ET + CB → LR) | default and tuned, ± SMOTE |
| Soft voting (XGB + ET + CB) | tuned, SMOTE (tests whether meta-learning beats plain averaging) |

**Tuning.** Optuna TPE ran 40/40/30 trials for XGBoost/ExtraTrees/CatBoost, with SMOTE `k_neighbors`
tuned jointly. The objective was the 5-fold CV PR-AUC on the training set. PR-AUC was used instead of F1@0.5
because the threshold is optimized separately. The meta LR (C, L1/L2) was then tuned on the OOF
meta-feature matrix. Chosen values are in `reports/metrics/tuning_results.json`. For example, CatBoost
used depth 3, 550 iterations, lr 0.013 and l2 12.4, and the meta LR used C = 1.30 with L2.

**Deployment choice.** The stacking + SMOTE architecture was fixed *a priori* by the project brief
(`selection.mode: fixed`) and was not chosen by looking at results. The best candidate by CV PR-AUC was
ExtraTrees without SMOTE (0.754 vs 0.743), and this is reported for transparency.

## 4. Results

### 4.1 Cross-validation (training set, 3×5 folds, mean ± std)

| Model | PR-AUC | ROC-AUC | F1@0.5 | Recall@0.5 | Brier | OOF F1 @ tuned thr |
|---|---|---|---|---|---|---|
| ExtraTrees, no SMOTE, tuned | **0.754** ± 0.057 | 0.845 | 0.585 | 0.471 | 0.162 | 0.689 |
| ExtraTrees, SMOTE, tuned | 0.747 ± 0.054 | 0.845 | 0.686 | 0.740 | 0.167 | 0.698 |
| CatBoost, no SMOTE, tuned | 0.745 ± 0.059 | 0.840 | 0.623 | 0.560 | 0.154 | 0.695 |
| Stacking, no SMOTE, tuned | 0.745 ± 0.062 | 0.841 | 0.625 | 0.564 | 0.155 | 0.692 |
| **Stacking, SMOTE, tuned (deployed)** | 0.743 ± 0.060 | 0.841 | 0.640 | 0.600 | 0.154 | 0.697 |
| Logistic Regression, no SMOTE | 0.741 ± 0.060 | 0.843 | 0.646 | 0.574 | **0.152** | 0.678 |
| CatBoost, SMOTE, tuned | 0.737 ± 0.061 | 0.837 | 0.689 | 0.756 | 0.162 | 0.700 |
| Soft voting, SMOTE, tuned | 0.733 ± 0.062 | 0.837 | 0.675 | 0.723 | 0.161 | 0.690 |
| XGBoost, SMOTE, tuned | 0.707 ± 0.064 | 0.821 | 0.661 | 0.689 | 0.171 | 0.670 |

All 21 candidates are listed in `reports/metrics/cv_summary.csv`. Tuning helped ExtraTrees, CatBoost and the stack
(stack PR-AUC 0.728 → 0.743). It did not help XGBoost under the evaluation CV: tuning-CV PR-AUC was 0.738, but
evaluation-CV PR-AUC was 0.707. That gap is selection optimism from tuning and evaluating on the same 614 patients.

### 4.2 Final test-set comparison (each model at its own CV-selected balanced threshold)

| Model | Thr | Acc | Prec | Recall | F1 | ROC-AUC | PR-AUC | Spec | Sens | Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| **Stacking, SMOTE, tuned (selected)** | 0.32 | 0.747 | 0.603 | 0.815 | **0.693** | 0.821 | 0.709 | 0.710 | 0.815 | 0.166 |
| XGBoost, no SMOTE, default | 0.27 | 0.734 | 0.582 | 0.852 | 0.692 | 0.823 | 0.701 | 0.670 | 0.852 | 0.162 |
| ExtraTrees, SMOTE, tuned | 0.31 | 0.734 | 0.584 | 0.833 | 0.687 | 0.825 | 0.707 | 0.680 | 0.833 | 0.164 |
| Stacking, no SMOTE, default | 0.26 | 0.727 | 0.575 | 0.852 | 0.687 | 0.818 | 0.679 | 0.660 | 0.852 | 0.170 |
| Soft voting, SMOTE, tuned | 0.31 | 0.734 | 0.587 | 0.815 | 0.682 | 0.821 | 0.707 | 0.690 | 0.815 | 0.164 |
| CatBoost, no SMOTE, tuned | 0.29 | 0.734 | 0.587 | 0.815 | 0.682 | 0.821 | 0.682 | 0.690 | 0.815 | 0.167 |
| Stacking, no SMOTE, tuned | 0.24 | 0.721 | 0.568 | 0.852 | 0.681 | 0.818 | 0.693 | 0.650 | 0.852 | 0.169 |
| XGBoost, SMOTE, tuned | 0.33 | 0.734 | 0.594 | 0.759 | 0.667 | 0.822 | 0.715 | 0.720 | 0.759 | 0.163 |
| CatBoost, SMOTE, tuned | 0.30 | 0.714 | 0.566 | 0.796 | 0.662 | 0.811 | 0.693 | 0.670 | 0.796 | 0.170 |
| Logistic Regression, no SMOTE | 0.37 | 0.734 | 0.614 | 0.648 | 0.631 | 0.813 | 0.673 | 0.780 | 0.648 | 0.175 |

The stack has the highest test F1, but by margins far smaller than the test-set uncertainty: the F1 CI
spans 0.60–0.78. Test ROC-AUC ranges only from 0.80 to 0.83 across all 21 candidates.

**Deployed model by operating mode (test).** Thresholds were chosen on training OOF predictions.

| Mode | Threshold | Precision | Recall | F1 | Specificity | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| High sensitivity | 0.25 | 0.566 | **0.870** | 0.686 | 0.640 | 47 | 36 | 7 | 64 |
| Balanced (default) | 0.32 | 0.603 | 0.815 | 0.693 | 0.710 | 44 | 29 | 10 | 71 |
| High precision | 0.63 | 0.686 | 0.444 | 0.539 | 0.890 | 24 | 11 | 30 | 89 |
| (naïve 0.5) | 0.50 | 0.635 | 0.611 | 0.623 | 0.810 | 33 | 19 | 21 | 81 |

The high-precision mode missed its precision target on the test set (0.686 vs the 0.75 target met on OOF data).
This is expected variance with 35 positive predictions.

## 5. Hypothesis checks

**Hypothesis 1.** SMOTE improves minority-class recall over the baseline models (by about 18 %).

| Measured | Value |
|---|---|
| CV recall@0.5 change from SMOTE (10 model pairs) | +3.4 to +26.9 pp; relative +6 % to +57 % |
| …for Random Forest (default) | +10.4 pp (**+18.0 % relative**) |
| …for the deployed stack (tuned) | +3.6 pp (+6.3 % relative) |
| CV PR-AUC change from SMOTE | −0.013 to +0.003 (no ranking improvement) |
| CV Brier change, single models | +0.003 to +0.010 (worse calibration) |
| Test F1 change at each model's tuned threshold | −2.5 to +3.1 pp (mixed) |

**Verdict: partially supported, and misleading as phrased.** SMOTE raises recall *at a fixed 0.5 threshold*,
and an ~18 % relative gain does occur for one baseline (Random Forest). But SMOTE does not make the models
rank diabetic patients better, and the same or higher recall comes from lowering the threshold. For example,
stacking *without* SMOTE at its tuned threshold reaches a test recall of 0.852. The effect on the final stack is small
(+6 % relative) because the meta-learner, trained on real-prevalence OOF data, largely undoes SMOTE's
probability shift.

**Hypothesis 2.** The stacking ensemble outperforms the individual classifiers in F1 (by 6–8 %) through ensemble
diversity and meta-learning.

| Comparison (stack vs …) | CV F1@0.5 (rel.) | OOF F1 at tuned thr | Test F1 at tuned thr (rel.) | Paired-bootstrap 95 % CI of test F1 diff | P(stack better) |
|---|---|---|---|---|---|
| XGBoost (SMOTE, tuned) | 0.640 vs 0.661 (−3.1 %) | 0.697 vs 0.670 | 0.693 vs 0.667 (+3.9 %) | [−0.023, +0.079] | 0.83 |
| ExtraTrees (SMOTE, tuned) | 0.640 vs 0.686 (−6.7 %) | 0.697 vs 0.698 | 0.693 vs 0.687 (+0.9 %) | [−0.037, +0.047] | 0.60 |
| CatBoost (SMOTE, tuned) | 0.640 vs 0.689 (−7.1 %) | 0.697 vs 0.700 | 0.693 vs 0.662 (+4.7 %) | [−0.008, +0.076] | 0.95 |
| Soft voting | 0.640 vs 0.675 (−5.1 %) | 0.697 vs 0.690 | 0.693 vs 0.682 (+1.6 %) | [−0.022, +0.044] | 0.73 |

On the 15 CV folds, the stack had a higher F1@0.5 than XGBoost / ExtraTrees / CatBoost in 3 / 1 / 1 folds, and
a higher PR-AUC in 14 / 6 / 11 folds. Across 5 seeds it beat every base learner on PR-AUC in 0 seeds.
Base-learner OOF probabilities correlate at **0.93–0.96**, so there is little diversity for meta-learning
to exploit.

**Verdict: not supported.** The 6–8 % F1 improvement was not observed. At 0.5, the stack's F1 is *lower*,
because the meta-learner restores real-prevalence probabilities and so predicts fewer positives. At tuned
thresholds, the stack is statistically indistinguishable from its best base learner.

**Summary:** *"Stacking matched the best single learner (test F1 0.693 vs 0.662–0.687; CIs
overlapping) while giving better-calibrated probabilities than SMOTE-trained single models (CV Brier 0.154 vs
0.162–0.171). Ablations showed SMOTE's recall gains at the default threshold are reproducible by threshold
tuning."*

## 6. Research questions

1. **Does SMOTE improve minority recall?** Yes at threshold 0.5 (see §5); no once thresholds are tuned. It does not improve PR-AUC or ROC-AUC.
2. **Does SMOTE improve F1?** At 0.5 it does in CV (+1.5 to +10.1 pp). At tuned thresholds the effect on test F1 is mixed and small.
3. **Does stacking outperform individual models?** Not significantly (see §5).
4. **Which base model contributes most?** The meta LR's standardized coefficients are CatBoost 0.64, ExtraTrees 0.61 and XGBoost 0.19 (raw: 2.17, 2.95, 0.57). In the drop-one ablation, CV PR-AUC changes by −0.009 without ExtraTrees, −0.005 without CatBoost and +0.0005 without XGBoost, so XGBoost is redundant.
5. **Does the ensemble improve calibration?** Compared with SMOTE singles, yes (CV Brier 0.154 vs 0.162–0.171; ECE 0.081 vs 0.106–0.140). Compared with logistic regression (0.152), no. Nested-CV calibration selection kept `none` for the stack because sigmoid (0.156) and isotonic (0.158) were worse. On the test set, Brier scores (0.162–0.175) are statistically indistinguishable.
6. **Best sensitivity/F1 threshold?** 0.32 maximizes OOF F1 (0.697, with recall 0.798 on OOF and 0.815 on test). Dropping to 0.25 raises test recall to 0.870 (7 missed diabetics instead of 10) at the cost of 7 more false positives, with F1 nearly unchanged (0.686). For screening-style use, 0.25 is the better trade-off.
7. **Which features dominate?** Glucose dominates (mean |SHAP| 0.147; permutation PR-AUC drop 0.163). Next come BMI (0.075 / 0.055), Age (0.047 / 0.063) and DiabetesPedigreeFunction (0.041 / 0.027). BloodPressure, Insulin and SkinThickness contribute little: permutation drops of 0.001, 0.001 and 0.006.
8. **Who is misclassified?** The training OOF analysis (n = 614) is consistent with the test set:
   - **False negatives.** 80 % of missed diabetics have glucose < 126 mg/dL, against 21 % of caught diabetics. Their medians are glucose 109 vs 152, age 28 vs 37.5, BMI 31.8 vs 34.9 and DPF 0.38 vs 0.51.
   - **False positives.** 59 % of false positives have glucose ≥ 126 (72 % on test), against 11 % of true negatives.
   - **The pattern.** The model leans heavily on glucose. Diabetic patients with normal-range OGTT glucose, who are young with moderate BMI, look like non-diabetics to it. SHAP on test FNs shows glucose pulling the probability down by an average of 0.098.
   - **Borderline cases.** Patients within ±0.10 of the threshold (109 OOF) have a 32 % error rate.
9. **Sensitivity to random seed.** Across 5 seeds × 5-fold CV, with both the split seed and the model seed varied, PR-AUC std is 0.005–0.015 per model (stack 0.733 ± 0.011, range 0.722–0.751). Model rankings flip between seeds.
10. **Stability across CV folds.** The stack's per-fold PR-AUC ranges from 0.642 to 0.857 (std 0.060), and F1@0.5 from 0.577 to 0.725. Differences between the top 8 candidates (≤ 0.02 PR-AUC) are well inside this range. With 614 training patients the data cannot separate these models.

## 7. Explainability

Exact SHAP values were computed on the calibrated probability output, with a background of 50 stratified training
patients. Additivity holds to 1e-15: base value 0.352 + Σ SHAP = model output. Global plots are in
`reports/figures/shap_*.png`. Per-patient explanations are returned by the API and UI. Explanations describe
**model behaviour, not clinical causation**. For example, "insulin lowered the estimate" means the model
associates that value with lower risk in this dataset; it is not a physiological claim.

## 8. Limitations

- **Small sample.** With 768 patients and 154 test cases, test-metric 95 % CIs are about ±0.08 wide.
- **Single, specific population.** The cohort is Pima Indian women aged 21–81, from 1965–1990s data, with no external validation.
- **Heavy missingness.** Insulin (49 %) and SkinThickness (30 %) were imputed with medians.
- **Label definition.** The outcome is an OGTT-based diagnosis, and Glucose *is* the OGTT measurement, so the strongest feature is close to the label definition.
- **Tuning optimism.** Hyperparameters were tuned on the same training data as the CV comparison (no nested CV), so CV scores of tuned models are optimistic. The test set is unaffected.
- **No fairness audit is possible.** There are no demographic attributes beyond age.
- **Not clinically validated.** This is not a diagnostic device and must not inform individual care.

## 9. Reproducibility

`python scripts/train.py` was executed twice from scratch. Both runs produced **bit-identical** CV metrics, tuned
hyperparameters and seed-stability results (max |difference| = 0.0). Library versions are pinned in
`requirements.txt` and recorded in `models/model_metadata.json`.
