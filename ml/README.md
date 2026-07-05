# Model Card — Cardiac Risk Random Forest (v2)

## Dataset

**Framingham Heart Study** (GitHub public mirror)

- **Source:** [Framingham Heart Study](https://github.com/GauravPadawe/Framingham-Heart-Study)
- **Samples:** 4,240 patients (4,189 after dropping 51 rows with missing values)
- **Target:** Binary — 10-year coronary heart disease (CHD) risk
  - 0 = no CHD within 10 years (84.9%)
  - 1 = CHD within 10 years (15.1%)
- **Fallback:** Cleveland Heart Disease dataset (297 samples) — used when
  Framingham URL is unreachable.

**Why Framingham instead of Cleveland?** The Cleveland dataset (used in v1)
predicted *diagnosed* heart disease from clinical measurements. Framingham
predicts *10-year future risk* from lifestyle + clinical factors — which is
closer to what the app does. It's also 14x larger.

## Features Used

### Core features (from the intake form):

| Form field          | Model feature | Description                    |
|---------------------|---------------|--------------------------------|
| age                 | age           | Age in years                   |
| biological_sex      | sex           | 1 = male, 0 = female          |
| resting_hr          | thalach       | Resting heart rate (bpm)       |
| systolic_bp         | trestbps      | Systolic blood pressure (mmHg) |
| total_cholesterol   | chol          | Serum cholesterol (mg/dL)      |

### Extra features (from Framingham, used when available):

| Feature         | Description                     |
|-----------------|---------------------------------|
| BMI             | Body mass index                 |
| glucose         | Serum glucose (mg/dL)           |
| diabetes        | Binary: 1 = diabetic            |
| currentSmoker   | Binary: 1 = current smoker      |
| cigsPerDay      | Cigarettes per day              |
| prevalentHyp    | Binary: 1 = hypertensive        |
| diaBP           | Diastolic blood pressure (mmHg) |

### Engineered features (derived):

| Feature            | Formula                          | Rationale                                 |
|--------------------|----------------------------------|-------------------------------------------|
| pulse_pressure     | trestbps - diaBP                 | Arterial stiffness marker                 |
| smoking_severity   | currentSmoker × cigsPerDay       | Captures dose beyond binary yes/no        |
| age_x_bp           | age × trestbps / 100             | High BP is worse at older ages            |
| age_x_bmi          | age × BMI / 100                  | Obesity is worse at older ages            |
| age_x_chol         | age × chol / 100                 | High cholesterol at older age amplifies risk |
| bmi_category       | 0-3 ordinal: underweight to obese | Non-linear BMI effects                   |

## Model Architecture

- **Algorithm:** Random Forest Classifier (`scikit-learn`)
- **Trees:** 300 (found optimal by GridSearchCV)
- **Max depth:** 5
- **Max features:** `sqrt`
- **Min samples split/leaf:** 5 / 2
- **Class weight:** Balanced
- **Random state:** 42 (fixed for reproducibility)

## Performance

### 10-fold stratified cross-validation

| Model Version          | Features | Accuracy (CV) | ROC-AUC (CV) |
|------------------------|:--------:|:-------------:|:------------:|
| **Default params**     | 18       | 0.682 ± 0.026 | 0.718 ± 0.042 |
| **Tuned (GridSearchCV)** | 18     | 0.678 ± 0.029 | 0.717 ± 0.042 |
| Default params (5-feat) | 5       | 0.689 ± 0.024 | 0.715 ± 0.041 |

### Key observations

1. **Adding more features (5 → 18) barely helped** — ROC-AUC went from
   0.715 to 0.718. The extra features (BMI, smoking, glucose, derived
   interactions) added information, but the model was already close to
   what 5 core features can achieve.

2. **Hyperparameter tuning didn't help** — the default Random Forest params
   (100 trees, depth 5) were already near-optimal for this data.

3. **The ceiling is ~0.72 ROC-AUC** — with self-reportable features alone
   (age, sex, BP, cholesterol, heart rate, BMI, smoking), you can't predict
   10-year cardiac risk much better than this. Clinical data (ECG, imaging,
   stress tests, genetic markers) would be needed to go higher.

4. **The simpler 5-feature model is almost as good** — for the app's
   purposes, the 5 core features (available on every form submission)
   capture ~99% of the signal. The extra features are optional
   improvements.

### Feature importance (tuned model, top 10)

| Rank | Feature         | Importance |
|------|-----------------|:----------:|
| 1    | age_x_bp        | 0.206      |
| 2    | age             | 0.123      |
| 3    | age_x_bmi       | 0.088      |
| 4    | trestbps        | 0.079      |
| 5    | age_x_chol      | 0.076      |
| 6    | pulse_pressure  | 0.064      |
| 7    | cigsPerDay      | 0.045      |
| 8    | smoking_severity| 0.045      |
| 9    | glucose         | 0.044      |
| 10   | diaBP           | 0.042      |

**Interpretation:** The age × BP interaction is the single strongest
predictor — the same blood pressure carries more risk at older ages. Raw
age itself is second. Most of the top features are age-related interactions,
confirming that age is the dominant risk factor and it amplifies the
impact of other factors.

## Limitations

1. **Modest accuracy.** ~0.72 ROC-AUC is meaningful (well above random
   0.50) but far from clinical-grade (0.90+). The model should not be used
   for real medical decisions.

2. **Class imbalance.** 15% disease prevalence means the naive baseline
   (always predict "no disease") achieves 85% accuracy. Our model's ~68%
   accuracy is below this baseline — accuracy is a misleading metric here.
   ROC-AUC (0.72) is the honest picture.

3. **Missing factors.** The model can't account for family history, genetic
   markers, ECG results, or lifestyle factors beyond smoking — all of which
   are strong predictors of cardiac risk. The rules-based profiler (Module 1)
   handles family history separately.

4. **Population.** Framingham is a predominantly white, US-based cohort
   studied since 1948. Results may not generalise to other populations.

5. **10-year risk, not a diagnosis.** The target is future risk, not current
   disease. A low risk score doesn't rule out existing conditions.

6. **Self-reported data.** The model is trained on clinical measurements,
   but the app uses self-reported values. Measurement error will reduce
   real-world performance further.
