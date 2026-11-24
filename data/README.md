# Data

## Source

**Pima Indians Diabetes Database** — originally from the US National Institute of Diabetes and
Digestive and Kidney Diseases (Smith et al., 1988, *Using the ADAP learning algorithm to forecast
the onset of diabetes mellitus*), distributed via the UCI Machine Learning Repository / Kaggle.

| | |
|---|---|
| File | `data/raw/diabetes.csv` (header-less UCI layout; the loader also accepts the Kaggle CSV with header) |
| Download | `python scripts/download_data.py` (URL in `configs/config.yaml`) |
| Version pin | SHA-256 `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af` — verified on every load |
| Rows / columns | 768 patients × 8 features + `Outcome` |
| Population | Female patients of Pima Indian heritage, aged ≥ 21, near Phoenix, Arizona |
| Target | `Outcome` = 1 if diabetes (WHO criteria: 2-hour post-load plasma glucose ≥ 200 mg/dL) was diagnosed within the study window |

| Column | Meaning | Unit |
|---|---|---|
| Pregnancies | Number of times pregnant | count |
| Glucose | Plasma glucose 2 h into an oral glucose tolerance test | mg/dL |
| BloodPressure | Diastolic blood pressure | mm Hg |
| SkinThickness | Triceps skin-fold thickness | mm |
| Insulin | 2-hour serum insulin | µU/mL |
| BMI | Body mass index | kg/m² |
| DiabetesPedigreeFunction | Family-history based genetic-risk score | — |
| Age | Age | years |

## Validation results (`reports/metrics/data_validation.json`)

- 768 rows, 9 columns, all numeric, **no duplicate rows**, **no explicit NaN values**.
- Target: 500 negative / 268 positive (**34.9 % positive**).
- No negative values; no values outside the plausibility limits in `src/data/validation.py`
  once zeros are treated as missing.

## Zero values: which are missing measurements?

| Column | Rows recorded as 0 | Share | Decision |
|---|---:|---:|---|
| Glucose | 5 | 0.7 % | **Missing** — a plasma glucose of 0 mg/dL is incompatible with life |
| BloodPressure | 35 | 4.6 % | **Missing** — a diastolic pressure of 0 is not a measurement of a living patient |
| SkinThickness | 227 | 29.6 % | **Missing** — a skin fold cannot be 0 mm |
| Insulin | 374 | 48.7 % | **Missing** — 0 µU/mL 2-h insulin is not physiologically plausible in this cohort; the rate matches "test not performed" |
| BMI | 11 | 1.4 % | **Missing** — BMI 0 implies zero body mass |
| Pregnancies | 111 | 14.5 % | **Genuine** — zero pregnancies is a valid count |
| DiabetesPedigreeFunction, Age | 0 | — | no zeros |

Zeros in SkinThickness and Insulin co-occur heavily (every row with SkinThickness = 0 also has
Insulin = 0), which is further evidence that they encode "not measured".

## Preprocessing decisions

1. **Zero → NaN** for the five columns above (`ZeroToNaN`, a stateless transformer).
2. **Median imputation** fitted on the training fold only (`SimpleImputer(strategy="median")`
   inside each pipeline, so every CV fold, every stacking inner fold and the final model compute
   their own medians from their own training rows). The median is robust to the strong right skew
   of Insulin and SkinThickness. Iterative/KNN imputation was not used in v1: with 49 % of Insulin
   missing, a model-based imputer would mostly propagate Glucose/BMI information while adding
   variance and complexity; this is listed as future work.
3. **Missing-indicator features** are available (`preprocessing.add_missing_indicators`) but
   disabled: missingness of Insulin/SkinThickness reflects the original study's data collection
   rather than patient physiology and would be unavailable in a deployment context.
4. **No outlier removal.** Tukey-fence outliers (e.g. Insulin > 360, DPF > 1.2) are plausible
   clinical values; tree models are insensitive to them and LR uses standardisation.
5. **Scaling** only for Logistic Regression (`StandardScaler`); tree ensembles use raw values.
6. **SMOTE** (where enabled) runs *after* imputation and *only on training folds*.
7. **Train/test split**: stratified 80/20 with seed 42 → `data/processed/train.csv` (614 rows)
   and `data/processed/test.csv` (154 rows), with original row ids preserved.

## Licence / usage

The dataset is widely redistributed for research and education (Kaggle lists it under CC0).
It is a historical research cohort and is not representative of other populations.
