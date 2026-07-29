# Model Card — Cardiac Risk Models (v3)

## Overview

CardioGenome now uses a **two-tier ML architecture**:

1. **Primary: NHANES XGBoost** — a gradient-boosted tree trained on CDC NHANES
   survey data (2017–2023). Better accuracy, more training data, more features.
2. **Fallback: Framingham Random Forest** — the original model, used if the
   NHANES model file is unavailable.

Both models predict a cardiac risk probability (0–100%) that feeds into the
hybrid scoring system (ML base + condition-specific rules boosts).

---

## Model 1: NHANES XGBoost *(primary)*

### Dataset

**NHANES 2017–2023** (CDC National Health and Nutrition Examination Survey)

- **Source:** [Kaggle — nhanes-cvd-raw-data-2017-23](https://www.kaggle.com/datasets/ahiduzzaman28/nhanes-cvd-raw-data-2017-23)
- **Samples:** 27,493 raw → 16,842 after dropping missing/refused CVD answers
  and filtering to adults (20–80 years)
- **Target:** Composite CVD — any self-reported diagnosis of congestive heart
  failure, coronary heart disease, heart attack, stroke, or angina
  - 0 = no CVD (87.8%)
  - 1 = CVD present (12.2%)
- **Population:** Multi-ethnic US nationally representative survey (vs
  Framingham's predominantly white Massachusetts cohort)
- **Timeframe:** 2017–2023 (vs Framingham's original 1948 cohort)

### Features Used

| Feature | In intake form? | Source | Imputed at inference? |
|---|---|---|---|
| **Age** (years) | ✅ Yes | Form field | No |
| **Systolic_BP** (mmHg) | ✅ Yes | Form field | No |
| **Total_Colesterol** (mg/dL) | ✅ Yes | Form field | No |
| BMI (kg/m²) | ❌ No | NHANES exam | Yes — median (28.2) |
| Diastolic_BP (mmHg) | ❌ No | NHANES exam | Yes — median (74) |
| Waist_circ (cm) | ❌ No | NHANES exam | Yes — median (99.1) |
| C_Reactive (mg/dL) | ❌ No | NHANES lab | Yes — median (0.21) |

**Note:** Features not collected in the form are imputed with training-set
medians at inference time. This is suboptimal but still allows the model to
use the 3 form-mapped features and benefit from the larger training set.

### Model Architecture

- **Algorithm:** XGBoost Classifier (`xgboost==3.3.0`)
- **Hyperparameters** (tuned via GridSearchCV, 108 combinations):
  - `n_estimators`: 100
  - `max_depth`: 4
  - `learning_rate`: 0.05
  - `subsample`: 0.7
  - `colsample_bytree`: 0.7
  - `scale_pos_weight`: auto (neg/pos ratio ≈ 7.2)
- **SMOTE oversampling** applied during training to handle class imbalance
- **Median imputation** for missing values (stored per-feature for inference)

### Performance

| Metric | Default params | Tuned (GridSearchCV) |
|---|---|---|
| **ROC-AUC** (10-fold CV) | 0.761 ± 0.018 | — |
| **ROC-AUC** (5-fold CV, tuning) | — | **0.795** |
| **ROC-AUC** (held-out test set) | 0.762 | **0.818** |
| **Accuracy** (held-out test set) | 0.611 | 0.392* |

*\*Accuracy drops with tuning because the model becomes more aggressive at
predicting CVD (higher recall, lower precision). With 12% prevalence,
accuracy is a misleading metric — ROC-AUC gives the honest picture.*

#### Comparison with previous model

| Model | Dataset | Samples | Features | ROC-AUC (CV) | ROC-AUC (test) |
|---|---|---|---|---|---|
| **XGBoost (tuned)** | NHANES 2017-23 | 16,842 | 7 | 0.761 ± 0.018 | **0.818** |
| Random Forest (tuned) | Framingham | 4,189 | 18 | 0.717 ± 0.042 | 0.715 |
| Random Forest (5-feat) | Framingham | 4,189 | 5 | 0.715 ± 0.041 | — |

**Improvement:** +0.10 ROC-AUC over the Framingham model.

### Feature Importance (tuned XGBoost)

| Rank | Feature | Importance |
|------|---------|:----------:|
| 1 | **Age** | **0.508** |
| 2 | Total_Colesterol | 0.100 |
| 3 | Diastolic_BP | 0.088 |
| 4 | Waist_circ | 0.085 |
| 5 | C_Reactive | 0.084 |
| 6 | Systolic_BP | 0.076 |
| 7 | BMI | 0.060 |

**Interpretation:** Age is overwhelmingly dominant (>50% of importance),
consistent with clinical knowledge. The 6 remaining features each contribute
6–10%, with cholesterol, blood pressure, and waist circumference being the
most informative of the modifiable factors.

---

## Model 2: Framingham Random Forest *(fallback)*

### Dataset

**Framingham Heart Study** (GitHub public mirror)

- **Source:** [Framingham Heart Study](https://github.com/GauravPadawe/Framingham-Heart-Study)
- **Samples:** 4,240 patients (4,189 after dropping 51 rows with missing values)
- **Target:** Binary — 10-year coronary heart disease (CHD) risk
  - 0 = no CHD within 10 years (84.9%)
  - 1 = CHD within 10 years (15.1%)

### Features Used

| Form field | Model feature | Description |
|---|---|---|
| age | age | Age in years |
| biological_sex | sex | 1 = male, 0 = female |
| resting_hr | thalach | Resting heart rate (bpm) |
| systolic_bp | trestbps | Systolic blood pressure (mmHg) |
| total_cholesterol | chol | Serum cholesterol (mg/dL) |

### Performance

| Metric | Value |
|---|---|
| **ROC-AUC** (10-fold CV) | 0.715 ± 0.041 |
| **Accuracy** (10-fold CV) | 0.689 ± 0.024 |

---

## How the hybrid system uses both models

```
User form data
     │
     ▼
NHANES XGBoost ──► Cardiac probability (0–100%)
    (primary)          │
                       │ × 0.4 = ML base score (0–40)
                       │
Framingham RF ◄────────┤ (fallback if NHANES missing)
                       │
                       ▼
           ┌──────────────────────┐
           │ Condition-specific   │
           │ rules boosts (0–60)  │
           │  • Family history    │
           │  • Personal symptoms │
           │  • Genetic variants  │
           └──────────────────────┘
                       │
                       ▼
              Hybrid score (0–100)
              per condition (HCM, LQTS, FH)
```

## Limitations (both models)

1. **Moderate accuracy.** ~0.82 ROC-AUC is meaningful but far from
   clinical-grade (0.90+). The model should not be used for real medical
   decisions.

2. **Imputed features.** The NHANES model uses median-imputed values for
   BMI, waist circumference, diastolic BP, and CRP — these are not actual
   patient measurements. A form collecting height/weight and waist
   circumference would unlock full feature usage.

3. **Self-reported data.** Both the NHANES target (CVD diagnosis) and
   Framingham target (CHD events) rely on self-report or clinical
   documentation. Measurement error reduces real-world performance.

4. **Population.** NHANES is more diverse than Framingham but still a US
   survey. Results may not generalise to non-US populations.

5. **Not a diagnosis.** The ML model predicts a probability, not a
   diagnosis. Low probability doesn't rule out existing conditions.

6. **Missing factors.** Neither model can account for genetic variants,
   detailed family history (beyond yes/no), ECG results, or imaging —
   all strong predictors the hybrid rules layer attempts to compensate for.
