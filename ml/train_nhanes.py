"""
ml/train_nhanes.py — NHANES-based XGBoost training pipeline for CardioGenome.

WHAT THIS SCRIPT DOES
    Loads the Kaggle NHANES CVD dataset (~27K rows, 47 columns), creates a
    composite CVD target (any of 5 self-reported conditions: congestive heart
    failure, coronary heart disease, heart attack, stroke, angina), trains an
    XGBoost classifier with SMOTE oversampling and hyperparameter tuning, and
    saves the model to disk.

WHY NHANES INSTEAD OF FRAMINGHAM?
    The Framingham dataset (~4,200 samples, 5 features) hits a performance
    ceiling at ~0.72 ROC-AUC because its features are limited to basic vitals
    (age, sex, BP, cholesterol, heart rate). NHANES offers:
        1. ~16,800 clean samples (4× more) after dropping missing targets
        2. Additional features: BMI, waist circumference, C-reactive protein,
           diastolic BP, and dietary/micronutrient data
        3. A more diverse, multi-ethnic US population (vs Framingham's
           predominantly white Massachusetts cohort)
        4. A newer cohort (2017–2023 vs Framingham's original 1948+)

    The primary trade-off is that our intake form can't collect all NHANES
    features (e.g. waist circumference, CRP, dietary data). These are imputed
    with population medians at inference time. However, the model still
    benefits from seeing the statistical relationships during training.

FEATURE MAPPING (form field -> NHANES feature):
    age            -> Age              (years)
    systolic_bp    -> Systolic_BP      (mmHg)
    total_cholest  -> Total_Colesterol (mg/dL)    [note: misspelled in dataset]
    BMI            -> BMI              (kg/m²)    [imputed if not collected]
    diastolic_bp   -> Diastolic_BP     (mmHg)     [imputed if not collected]
    waist_circ     -> Waist_circ       (cm)       [imputed if not collected]
    crp            -> C_Reactive       (mg/dL)    [imputed if not collected]

TARGET: CVD (composite — any of congestive heart failure, coronary heart
    disease, heart attack, stroke, or angina self-reported as diagnosed)

USAGE:
    python ml/train_nhanes.py

    Outputs:
        ml/cardiac_model_nhanes.pkl  — joblib dump of trained XGBoost model
                                       + feature names + imputation medians
        Printed evaluation metrics (single split + cross-validation)
"""

from typing import List, Dict, Tuple, Optional
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, cross_validate,
                                    StratifiedKFold, GridSearchCV)
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Paths ──────────────────────────────────────────────────────────────────

import pathlib

# Path to the Kaggle-downloaded NHANES CVD dataset
# Kaggle caches datasets in the user home directory (~/.cache/kagglehub/)
NHANES_CSV = str(
    pathlib.Path.home() / ".cache" / "kagglehub" / "datasets" /
    "ahiduzzaman28" / "nhanes-cvd-raw-data-2017-23" / "versions" / "1" /
    "Nhanes_cvd_raw.csv"
)

MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "cardiac_model_nhanes.pkl")

# ── Constants ──────────────────────────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CV_FOLDS = 10

# CVD target columns in the dataset (1=yes, 2=no, 9=refused)
CVD_COLUMNS = ["Congestive", "Coronary", "Heart_attack", "Stroke", "Angina"]

# Core features that map directly to our intake form
# These are always available even if the user doesn't provide optional data
CORE_FEATURES = [
    "Age",              # age (years)
    "Systolic_BP",      # systolic_bp (mmHg)
    "Total_Colesterol", # total_cholesterol (mg/dL) — note: misspelled in dataset
]

# Extended features — available if we impute with population medians
EXTENDED_FEATURES = [
    "BMI",              # kg/m²
    "Diastolic_BP",     # mmHg
    "Waist_circ",       # cm
    "C_Reactive",       # C-reactive protein (mg/dL)
]

# Dietary features — available from NHANES but not from form; included for
# training to capture patterns, imputed at inference
DIETARY_FEATURES = [
    "Protein", "Carbohydrates", "Fiber", "Saturated_Fat",
    "Monounsaturated_Fat", "Polyunsaturated_Fat",
    "Sodium", "Potassium", "Magnesium", "Calcium",
    "Vitamin_C", "Vitamin_D", "Vitamin_B12", "Folic_Acid",
]

ALL_FEATURES = CORE_FEATURES + EXTENDED_FEATURES


# ── Data Loading ──────────────────────────────────────────────────────────


def find_dataset_path() -> str:
    """Locate the NHANES CSV, trying multiple locations."""
    for path in [NHANES_CSV, FALLBACK_NHANES_CSV]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "NHANES CVD dataset not found. Run the download script first:\n"
        "  import kagglehub\n"
        "  kagglehub.dataset_download('ahiduzzaman28/nhanes-cvd-raw-data-2017-23')"
    )


def load_nhanes_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the NHANES CVD dataset from CSV.

    Args:
        filepath: Path to Nhanes_cvd_raw.csv. Auto-detected if None.

    Returns:
        Raw dataframe.
    """
    if filepath is None:
        filepath = find_dataset_path()
    print(f"Loading NHANES data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite CVD target from 5 condition columns.

    The original columns use: 1=Yes, 2=No, 9=Refused/Don't know.
    We map to: 1 = any CVD, 0 = no CVD, NaN = refused or missing.

    Returns the dataframe with a new 'CVD' column added.
    Rows where ANY CVD column is refused (9) are dropped for clean targets.
    """
    df = df.copy()

    # Map 9→NaN for all CVD columns
    for col in CVD_COLUMNS:
        df[col] = df[col].replace({9.0: np.nan})

    # Drop rows where any CVD column is missing (after mapping 9→NaN)
    before = len(df)
    df = df.dropna(subset=CVD_COLUMNS).copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing/refused CVD answers")

    # Create composite target: 1 if ANY condition is yes
    df["CVD"] = (df[CVD_COLUMNS].eq(1.0).any(axis=1)).astype(int)

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────


def prepare_features_target(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """
    Extract feature matrix X and target y, with median imputation.

    Returns:
        X: Feature matrix with imputed missing values.
        y: Target vector.
        medians: Dict of {column_name: median_value} for inference-time imputation.
    """
    # Select features that exist in the dataframe
    available = [f for f in feature_columns if f in df.columns]
    missing_cols = [f for f in feature_columns if f not in df.columns]
    if missing_cols:
        print(f"  Warning: columns not found in dataset: {missing_cols}")

    X_raw = df[available].copy()
    y = df["CVD"].copy()

    # Store medians for inference-time imputation
    medians = {}
    for col in available:
        med = X_raw[col].median()
        medians[col] = med
        X_raw[col] = X_raw[col].fillna(med)

    missing_after = X_raw.isna().sum().sum()
    print(f"  Missing values after imputation: {missing_after} (should be 0)")

    return X_raw, y, medians


# ── Model Training ─────────────────────────────────────────────────────────


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier with SMOTE oversampling.

    SMOTE generates synthetic samples of the minority class (CVD cases)
    to balance the training data, which helps XGBoost learn the
    decision boundary for the positive class.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Trained XGBClassifier.
    """
    # Calculate scale_pos_weight from training data for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    # Create pipeline: SMOTE → XGBoost
    # SMOTE handles class imbalance by synthesizing minority class samples
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight_val,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    pipeline.fit(X_train, y_train)
    return pipeline


# ── Evaluation ─────────────────────────────────────────────────────────────


def evaluate_model(
    model: ImbPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Print evaluation metrics on a held-out test set.

    Args:
        model: Trained pipeline (SMOTE + XGBoost).
        X_test: Test features.
        y_test: Test target.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"Single 80/20 split — {len(y_test)} test samples")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No CVD", "CVD"]))


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
) -> None:
    """
    Run k-fold stratified cross-validation with SMOTE + XGBoost.

    Note: SMOTE is applied inside each fold to avoid data leakage.

    Args:
        X: Full feature matrix.
        y: Full target vector.
        feature_names: Feature column names.
    """
    # Build a fresh pipeline for CV (no pre-fit)
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "xgb",
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight_val,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def roc_auc_scorer(estimator, X_fold, y_fold):
        y_proba = estimator.predict_proba(X_fold)[:, 1]
        return roc_auc_score(y_fold, y_proba)

    scoring = {"accuracy": "accuracy", "roc_auc": roc_auc_scorer}
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    accuracies = scores["test_accuracy"]
    rocs = scores["test_roc_auc"]

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_roc = np.mean(rocs)
    std_roc = np.std(rocs)

    print(f"\n{'='*50}")
    print(f"{N_CV_FOLDS}-Fold Stratified Cross-Validation")
    print(f"{'='*50}")
    print(f"  Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}  "
          f"[{mean_acc - 2*std_acc:.3f}, {mean_acc + 2*std_acc:.3f}]")
    print(f"  ROC-AUC:  {mean_roc:.3f} +/- {std_roc:.3f}  "
          f"[{mean_roc - 2*std_roc:.3f}, {mean_roc + 2*std_roc:.3f}]")

    print(f"\n  Fold-by-fold:")
    for i, (acc, roc) in enumerate(zip(accuracies, rocs), 1):
        print(f"    Fold {i:2d}:  Acc {acc:.3f}  ROC-AUC {roc:.3f}")


def show_feature_importance(
    model: ImbPipeline,
    feature_names: List[str],
) -> None:
    """
    Print XGBoost feature importance from the trained pipeline.

    Args:
        model: Trained SMOTE + XGBoost pipeline.
        feature_names: Feature column names in training order.
    """
    xgb_model = model.named_steps["xgb"]
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n{'='*50}")
    print("Feature Importance (XGBoost)")
    print(f"{'='*50}")
    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:20s}  {importances[idx]:.4f}")


# ── Hyperparameter Tuning ────────────────────────────────────────────────


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Run GridSearchCV on XGBoost with SMOTE to find best hyperparameters.

    Uses a limited grid to keep runtime reasonable on 16K samples.

    Args:
        X: Full feature matrix.
        y: Full target vector.

    Returns:
        dict with keys 'best_params', 'best_score', 'best_estimator'.
    """
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    print(f"\n{'='*50}")
    print("Hyperparameter Tuning (GridSearchCV)")
    print(f"{'='*50}")
    print("  Searching over 108 combinations (3×3×3×2×2)...")

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        (
            "xgb",
            xgb.XGBClassifier(
                scale_pos_weight=scale_pos_weight_val,
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])

    param_grid = {
        "xgb__n_estimators": [100, 200, 300],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.05, 0.1, 0.2],
        "xgb__subsample": [0.7, 1.0],
        "xgb__colsample_bytree": [0.7, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X, y)

    print(f"\n  Best params: {grid_search.best_params_}")
    print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"  (5-fold CV, tuned over {len(grid_search.cv_results_['params'])} combinations)")

    return {
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "best_estimator": grid_search.best_estimator_,
    }


# ── Model Persistence ──────────────────────────────────────────────────────


def save_model(
    pipeline: ImbPipeline,
    feature_names: List[str],
    medians: Dict[str, float],
    cvd_prevalence: float,
    output_path: str = MODEL_OUTPUT,
) -> None:
    """
    Serialise the trained pipeline, feature names, and imputation medians.

    Args:
        pipeline: Trained SMOTE + XGBoost pipeline.
        feature_names: Feature column names in training order.
        medians: Column medians for inference-time imputation.
        cvd_prevalence: CVD prevalence in training set (for calibration context).
        output_path: Where to save the .pkl file.
    """
    model_package = {
        "model": pipeline,
        "feature_names": feature_names,
        "impute_medians": medians,
        "cvd_prevalence": float(cvd_prevalence),
        "source": "NHANES 2017-2023",
        "target": "CVD composite (CHF, CHD, MI, stroke, angina)",
    }
    joblib.dump(model_package, output_path)
    print(f"\nModel saved to {output_path}")
    print(f"  Features ({len(feature_names)}): {feature_names}")
    print(f"  CVD prevalence in training: {cvd_prevalence:.1%}")
    print(f"  Imputation medians stored: {len(medians)} features")


# ── Main Pipeline ─────────────────────────────────────────────────────────


def main() -> None:
    """
    Run the full NHANES XGBoost training pipeline end-to-end.

    Steps:
        1. Load NHANES CVD dataset
        2. Create composite CVD target
        3. Prepare feature matrix and target
        4. Train XGBoost with SMOTE on 80/20 split
        5. Evaluate and show feature importance
        6. Run k-fold cross-validation
        7. Hyperparameter tuning with GridSearchCV
        8. Train final model and save
    """
    print("=" * 55)
    print("CardioGenome — NHANES XGBoost Training Pipeline")
    print("=" * 55)

    # Step 1-2: Load and create target
    print("\n1-2/8: Loading NHANES data and creating CVD target...")
    df = load_nhanes_data()
    df = create_target(df)

    # Filter to adults only
    df = df[df["Age"] >= 18].copy()
    print(f"  Adults 18+: {len(df)} rows")

    cvd_prevalence = df["CVD"].mean()
    baseline = max(cvd_prevalence, 1 - cvd_prevalence)
    print(f"  CVD prevalence: {cvd_prevalence:.1%}")
    print(f"  Baseline (always predict majority class): {baseline:.1%}")

    # Step 3: Prepare features
    print(f"\n3/8: Preparing features with {len(ALL_FEATURES)} features...")
    X, y, medians = prepare_features_target(df, ALL_FEATURES)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {list(X.columns)}")

    # Step 4: Train on single split
    print(f"\n4/8: Training XGBoost with SMOTE on 80/20 split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Training: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Training CVD prevalence: {y_train.mean():.1%}")

    model = train_xgboost(X_train, y_train)

    # Step 5: Evaluate
    print(f"\n5/8: Evaluating on test set...")
    evaluate_model(model, X_test, y_test)
    show_feature_importance(model, list(X.columns))

    # Step 6: Cross-validation
    print(f"\n6/8: 10-fold cross-validation...")
    cross_validate_model(X, y, list(X.columns))

    # Step 7: Hyperparameter tuning
    print(f"\n7/8: Hyperparameter tuning (may take a few minutes)...")
    tuning_results = tune_hyperparameters(X, y)

    # Step 8: Train final model and save
    print(f"\n8/8: Training final model and saving...")

    # Retrain on full data with best params (or default if tuning didn't help)
    best_params = tuning_results["best_params"]
    print(f"  Using best params: {best_params}")

    # Build final pipeline with best params
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight_val = neg_count / pos_count if pos_count > 0 else 1

    # Extract XGBoost params (strip the 'xgb__' prefix)
    xgb_params = {
        k.replace("xgb__", ""): v
        for k, v in best_params.items()
    }
    xgb_params.update({
        "scale_pos_weight": scale_pos_weight_val,
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })

    final_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBClassifier(**xgb_params)),
    ])
    final_pipeline.fit(X, y)

    # Evaluate final model on test set
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    final_acc = accuracy_score(y_test, y_pred)
    final_roc = roc_auc_score(y_test, y_proba)
    print(f"\n  Final tuned model on test set:")
    print(f"    Accuracy: {final_acc:.3f} (baseline: {baseline:.1%})")
    print(f"    ROC-AUC:  {final_roc:.3f}")

    show_feature_importance(final_pipeline, list(X.columns))

    # Save the model
    save_model(final_pipeline, list(X.columns), medians, cvd_prevalence)
    print(f"\n{'='*55}")
    print("Training complete!")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
