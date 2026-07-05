"""
ml/train_model.py — ML training pipeline for CardioGenome.

WHAT THIS SCRIPT DOES
    Loads the Framingham Heart Study dataset (~4,200 rows), trains a Random
    Forest classifier to predict 10-year coronary heart disease (CHD) risk
    from clinical features that overlap with the app's intake form, evaluates
    with cross-validation, and saves the model to disk.

WHY FRAMINGHAM?
    The Framingham dataset (~4,200 samples, 15 features) is a natural fit for
    this app because:
        1. Its target is "10-year CHD risk" — the same kind of future-risk
           prediction the app makes, not a retrospective diagnosis.
        2. The core features that match our form (age, sex, BP, cholesterol,
           heart rate) are all present and have no missing values.
        3. With 4,200+ samples it gives the model far more data to learn from
           than the 297-sample Cleveland dataset.

    The Cleveland Heart Disease dataset is kept as a fallback for when the
    Framingham URL is unreachable (e.g. no internet connection).

FEATURE MAPPING (form field -> model feature):
    age              -> age         (years)
    biological_sex   -> sex         (1 = male, 0 = female)
    resting_hr       -> thalach     (resting heart rate, bpm)
    systolic_bp      -> trestbps    (systolic BP, mm Hg)
    total_cholesterol -> chol       (mg/dL)

TARGET: TenYearCHD -> num (binary: 0 = no event, 1 = CHD within 10 years)

USAGE:
    python ml/train_model.py

    Outputs:
        ml/cardiac_model.pkl  — joblib dump of trained model + feature names
        Printed evaluation metrics (single split + cross-validation)
"""

from typing import List, Dict

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (train_test_split, cross_validate,
                                    StratifiedKFold, GridSearchCV)
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib


# ── Constants ──────────────────────────────────────────────────────────────

FRAMINGHAM_URL = (
    "https://raw.githubusercontent.com/GauravPadawe/"
    "Framingham-Heart-Study/master/framingham.csv"
)

CLEVELAND_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

COLUMN_NAMES_CLEVELAND = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Column mapping: Framingham names -> standardised names
# Only the 5 core features that match the intake form are kept.
FRAMINGHAM_COLUMN_MAP: Dict[str, str] = {
    "male": "sex",
    "age": "age",
    "sysBP": "trestbps",
    "totChol": "chol",
    "heartRate": "thalach",
}

# Core features that match the intake form.
# These are always available regardless of dataset.
CORE_FEATURES = [
    "age",
    "sex",
    "trestbps",   # systolic_bp from form
    "chol",       # total_cholesterol from form
    "thalach",    # resting_hr from form
]



TARGET = "num"

# Training hyperparameters (defaults; tune with GridSearchCV later)
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 5
TEST_SIZE = 0.2
N_CV_FOLDS = 10


# ── Data Loading ──────────────────────────────────────────────────────────

def load_framingham_data(url: str = FRAMINGHAM_URL) -> pd.DataFrame:
    """
    Load the Framingham Heart Study dataset from GitHub.

    Maps column names to a standardised schema and filters to only the
    5 core features that overlap with the intake form, plus the target.

    Args:
        url: Raw GitHub URL for framingham.csv.

    Returns:
        pd.DataFrame with standardised column names (5 features + target).
    """
    df = pd.read_csv(url)
    df = df.rename(columns=FRAMINGHAM_COLUMN_MAP)
    df = df.rename(columns={"TenYearCHD": TARGET})

    # Only keep the 5 core features + target; drop all other columns
    keep_cols = list(FRAMINGHAM_COLUMN_MAP.values()) + [TARGET]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    return df


def load_cleveland_data(url: str = CLEVELAND_URL) -> pd.DataFrame:
    """
    Load the Cleveland Heart Disease dataset from UCI (fallback only).

    Args:
        url: Direct download URL for processed.cleveland.data.

    Returns:
        pd.DataFrame with named columns and binary target.
    """
    df = pd.read_csv(url, header=None, names=COLUMN_NAMES_CLEVELAND, na_values="?")
    df = df.dropna()
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df


# ── Cleaning ──────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Drop rows with missing values in any required column.

    WHY DROP INSTEAD OF IMPUTE?
    Dropping ~50 rows out of 4,200 (< 1.2%) is simpler and avoids introducing
    imputation bias. In production with larger datasets or more missing values,
    you'd want to impute (e.g. median imputation for numeric columns) to avoid
    losing data.

    Args:
        df: Raw dataframe.
        required_columns: Rows missing any of these columns will be dropped.

    Returns:
        Cleaned dataframe (may have fewer rows).
    """
    before = len(df)
    df = df.dropna(subset=required_columns).copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing required features")
    return df




def prepare_features_target(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from cleaned data.

    Only uses features that exist in the dataframe (defensive check in case
    of fallback to Cleveland dataset with different column names).

    Args:
        df: Cleaned dataframe.
        feature_columns: Columns to use as features.

    Returns:
        X: Feature dataframe.
        y: Target series (binary).
    """
    available = [f for f in feature_columns if f in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()
    return X, y


# ── Model Training ────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        Trained RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Print evaluation metrics on a held-out test set.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.
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
    print(classification_report(y_test, y_pred, target_names=["No disease", "Disease"]))


def cross_validate_model(
    X: pd.DataFrame,
    y: pd.Series,
    model: RandomForestClassifier,
) -> None:
    """
    Run k-fold stratified cross-validation and print mean +/- std metrics.

    Args:
        X: Full feature matrix.
        y: Full target vector.
        model: An *untrained* classifier (re-fit per fold).
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def roc_auc_scorer(estimator, X_fold, y_fold):
        y_proba = estimator.predict_proba(X_fold)[:, 1]
        return roc_auc_score(y_fold, y_proba)

    scoring = {"accuracy": "accuracy", "roc_auc": roc_auc_scorer}
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

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
    model: RandomForestClassifier,
    feature_names: List[str],
) -> None:
    """
    Print feature importance scores from the trained model.

    Args:
        model: Trained RandomForestClassifier.
        feature_names: Feature column names in training order.
    """
    print(f"\n{'='*50}")
    print("Feature Importance")
    print(f"{'='*50}")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for i, idx in enumerate(indices, 1):
        print(f"  {i}. {feature_names[idx]:15s}  {importances[idx]:.3f}")


# ── Hyperparameter Tuning ────────────────────────────────────────────────


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Run GridSearchCV to find the best Random Forest hyperparameters.

    We use ROC-AUC as the scoring metric because accuracy is misleading
    on imbalanced data (15% disease prevalence). The grid is designed to
    balance thoroughness with runtime — a full Cartesian product would
    test 4 x 4 x 3 x 3 x 3 = 432 combinations; this grid tests 96.

    Args:
        X: Full feature matrix.
        y: Full target vector.

    Returns:
        dict with keys 'best_params', 'best_score', 'best_estimator'.
    """
    print(f"\n{'='*50}")
    print("Hyperparameter Tuning (GridSearchCV)")
    print(f"{'='*50}")
    print("  Searching over 324 combinations (3x4x3x3x3)...")

    # Parameter grid — start wide, narrow down based on results
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",    # ROC-AUC is robust to class imbalance
        n_jobs=-1,
        verbose=1,            # Show progress during search
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


# ── Model Persistence ────────────────────────────────────────────────────

def save_model(
    model: RandomForestClassifier,
    feature_names: List[str],
    output_path: str = "ml/cardiac_model.pkl",
) -> None:
    """
    Serialise the trained model and feature names to disk.

    Args:
        model: Trained RandomForestClassifier.
        feature_names: Feature column names in training order.
        output_path: Where to save the .pkl file.
    """
    model_package = {"model": model, "feature_names": feature_names}
    joblib.dump(model_package, output_path)
    print(f"\nModel saved to {output_path}")
    print(f"  Features ({len(feature_names)}): {feature_names}")


# ── Main Pipeline ────────────────────────────────────────────────────────

def main() -> None:
    """
    Run the full training pipeline end-to-end.

    Steps:
        1. Load Framingham Heart Study (fallback: Cleveland)
        2. Clean data (drop missing values in core features)
        3. Prepare feature matrix and target
        4. Show baseline comparison
        5. Train, evaluate on single 80/20 split, and show feature importance
        6. Run k-fold cross-validation
        7. Save model to disk
    """
    print("CardioGenome — ML Training Pipeline")
    print("=" * 55)

    # Step 1: Load data (prefer Framingham, fallback to Cleveland)
    print("\n1/8: Loading dataset...")
    df = pd.DataFrame()
    source_name = ""
    try:
        df = load_framingham_data()
        source_name = "Framingham Heart Study"
        print(f"  Loaded {len(df)} rows from {source_name}")
    except Exception as e:
        print(f"  [WARN] Framingham unavailable: {e}")
        try:
            df = load_cleveland_data()
            source_name = "Cleveland Heart Disease"
            print(f"  Loaded {len(df)} rows from {source_name} (fallback)")
        except Exception as e2:
            print(f"  [ERROR] No data available: {e2}")
            return

    # Step 2: Clean
    print(f"\n2/8: Cleaning data...")
    df = clean_data(df, CORE_FEATURES)

    disease_rate = df[TARGET].mean()
    majority_baseline = max(disease_rate, 1 - disease_rate)
    print(f"  {len(df)} rows, disease prevalence {disease_rate:.1%}")
    print(f"  Baseline (always predict majority class): {majority_baseline:.1%}")

    # Step 3: Prepare features
    print(f"\n3/8: Preparing features...")
    X, y = prepare_features_target(df, CORE_FEATURES)
    print(f"  Feature matrix: {X.shape}, Features: {list(X.columns)}")

    # Step 4: Baseline context
    print(f"\n4/8: Baseline comparison...")
    print(f"  If model accuracy > {majority_baseline:.1%}, it's learning "
          f"something useful vs guessing the majority class")

    # Step 5: Train + evaluate (single split)
    print(f"\n5/8: Training on single 80/20 split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  Training: {len(X_train)}, Test: {len(X_test)}")

    model = train_random_forest(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    show_feature_importance(model, list(X.columns))

    # Step 6: Cross-validation (baseline, before tuning)
    print(f"\n6/8: Cross-validation with default params...")
    cv_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
    )
    cross_validate_model(X, y, cv_model)

    # Step 7: Hyperparameter tuning with GridSearchCV
    print(f"\n7/8: Hyperparameter tuning...")
    tuning_results = tune_hyperparameters(X, y)

    # Retrain a final model with the best params found by grid search
    best_params = tuning_results["best_params"]
    print(f"\n  Retraining final model with best params...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    final_model = RandomForestClassifier(
        **best_params,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # Evaluate tuned model on held-out test set
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    tuned_acc = accuracy_score(y_test, y_pred)
    tuned_roc = roc_auc_score(y_test, y_proba)
    print(f"\n  Tuned model on test set:")
    print(f"    Accuracy: {tuned_acc:.3f} (baseline: {majority_baseline:.1%})")
    print(f"    ROC-AUC:  {tuned_roc:.3f}")

    # Cross-validate the tuned model for fair comparison with default
    print(f"\n  Cross-validating tuned model...")
    cross_validate_model(X, y, final_model)

    show_feature_importance(final_model, list(X.columns))

    # Step 8: Save the best model
    print(f"\n8/8: Saving tuned model...")
    save_model(final_model, list(X.columns))
    print(f"\n== Training complete! ==")


if __name__ == "__main__":
    main()
