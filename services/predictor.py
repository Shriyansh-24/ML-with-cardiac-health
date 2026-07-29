"""
services/predictor.py — ML + rules hybrid predictor for CardioGenome.

WHAT THIS MODULE DOES
    Combines a machine learning model (either the NHANES XGBoost model or
    the Framingham Random Forest fallback) with rules-based risk factors
    that the model was never trained on (family history, genetic variants,
    personal symptoms).

MODEL ARCHITECTURE (two-tier):
    1. **Primary: NHANES XGBoost** (ml/cardiac_model_nhanes.pkl)
       - Trained on NHANES 2017-2023 (~16,800 samples, 7 features)
       - Target: composite CVD (CHF, CHD, MI, stroke, angina)
       - ROC-AUC: 0.818 (tuned), 0.761 ± 0.018 (10-fold CV)
       - Uses Age, Systolic_BP, Total_Cholesterol from form + imputed features

    2. **Fallback: Framingham Random Forest** (ml/cardiac_model.pkl)
       - Trained on Framingham Heart Study (~4,200 samples, 5 features)
       - Target: 10-year CHD risk
       - ROC-AUC: 0.715 ± 0.041 (10-fold CV)
       - Uses age, sex, BP, cholesterol, heart rate from form

    Both produce a cardiac risk probability (0.0-1.0). The hybrid layer
    then adds condition-specific boosts (0-60 points) for factors the
    ML model never saw: family history, genetic variants, symptoms.

WHY TWO MODELS?
    The NHANES model is strictly better (higher accuracy, more features,
    more training data). However, it relies on median-imputed features
    (BMI, waist circumference, CRP) that we can't collect from the form.
    The Framingham model uses only form-available features and serves as
    a reliable fallback if the NHANES model file is missing.
"""

from typing import List, Optional, Dict, Any
import os
import joblib
import pandas as pd

from services.risk_profiler import UserHealthData


# ── Model Paths ────────────────────────────────────────────────────────────

# Primary: NHANES XGBoost (better accuracy, uses imputed features)
NHANES_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "cardiac_model_nhanes.pkl")
)

# Fallback: Framingham Random Forest (uses only form-available features)
FRAMINGHAM_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "cardiac_model.pkl")
)

# How many points the ML model can contribute (out of 100)
ML_MAX_SCORE = 40

# Risk-level thresholds for the raw cardiac probability
# These work for both NHANES (CVD probability) and Framingham (CHD probability)
# since both output values in roughly similar ranges.
RISK_TIERS = [
    (0.06, "Low"),
    (0.12, "Low-Moderate"),
    (0.18, "Moderate"),
    (0.28, "Elevated"),
    (float("inf"), "High"),
]

# ── Model Loading (lazy, two-tier) ────────────────────────────────────────

_model_package = None       # module-level cache; loaded once on first request
_model_source = None        # "nhanes" or "framingham"


def _ensure_model_loaded() -> str:
    """
    Load the best available model from disk on first call.

    Tries NHANES XGBoost first, then falls back to Framingham Random Forest.

    Returns:
        str: Source of the loaded model ("nhanes" or "framingham").
    """
    global _model_package, _model_source
    if _model_package is not None:
        return _model_source

    if os.path.exists(NHANES_MODEL_PATH):
        _model_package = joblib.load(NHANES_MODEL_PATH)
        _model_source = "nhanes"
        return _model_source

    if os.path.exists(FRAMINGHAM_MODEL_PATH):
        _model_package = joblib.load(FRAMINGHAM_MODEL_PATH)
        _model_source = "framingham"
        return _model_source

    raise FileNotFoundError(
        f"No model found. Tried:\n"
        f"  1. {NHANES_MODEL_PATH} (NHANES XGBoost)\n"
        f"  2. {FRAMINGHAM_MODEL_PATH} (Framingham Random Forest)\n"
        f"Run `python ml/train_nhanes.py` or `python ml/train_model.py` first."
    )


def model_available() -> bool:
    """Check if any trained model exists on disk."""
    return os.path.exists(NHANES_MODEL_PATH) or os.path.exists(FRAMINGHAM_MODEL_PATH)


def get_model_source() -> str:
    """
    Return which model is actively loaded ("nhanes", "framingham", or "none").
    """
    global _model_source
    if _model_source is not None:
        return _model_source
    try:
        return _ensure_model_loaded()
    except FileNotFoundError:
        return "none"


# ── Feature Preparation ────────────────────────────────────────────────────


def _prepare_features_nhanes(data: UserHealthData) -> pd.DataFrame:
    """
    Prepare features for the NHANES XGBoost model.

    Maps form fields to NHANES feature names. Uses stored median values
    to impute features the form doesn't collect (BMI, waist circumference,
    diastolic BP, C-reactive protein).

    Args:
        data: Parsed user health data from the intake form.

    Returns:
        pd.DataFrame with exactly 1 row and the model's feature columns.
    """
    impute_medians = _model_package.get("impute_medians", {})
    feature_names = _model_package.get("feature_names", [])

    # Build features dict with form-available values
    features: Dict[str, float] = {
        "Age": float(data["age"]),
        "Systolic_BP": float(data["systolic_bp"]),
        "Total_Colesterol": float(data["total_cholesterol"]),
    }

    # Impute missing features with training-set medians
    for feat_name in feature_names:
        if feat_name not in features:
            features[feat_name] = float(impute_medians.get(feat_name, 0.0))

    # Ensure features are in the exact order the model expects
    ordered = {fn: features[fn] for fn in feature_names}
    return pd.DataFrame([ordered])


def _prepare_features_framingham(data: UserHealthData) -> pd.DataFrame:
    """
    Prepare features for the Framingham Random Forest model.

    The model was trained on ['age', 'sex', 'trestbps', 'chol', 'thalach'].
    Maps form fields and converts sex to binary.

    Args:
        data: Parsed user health data from the intake form.

    Returns:
        pd.DataFrame with exactly 1 row and the 5 model features.
    """
    sex = 1 if data["biological_sex"] == "male" else 0

    features = {
        "age": data["age"],
        "sex": sex,
        "trestbps": data["systolic_bp"],
        "chol": data["total_cholesterol"],
        "thalach": data["resting_hr"],
    }

    return pd.DataFrame([features])


# ── ML Prediction ──────────────────────────────────────────────────────────


def _get_cardiac_probability(data: UserHealthData) -> float:
    """
    Run the loaded ML model on the user's data.

    Routes to the correct feature-preparation function based on which
    model is loaded (NHANES or Framingham).

    Args:
        data: Parsed user health data.

    Returns:
        float: Predicted cardiac risk probability (0.0 - 1.0).
    """
    source = _ensure_model_loaded()

    if source == "nhanes":
        X = _prepare_features_nhanes(data)
    else:
        X = _prepare_features_framingham(data)

    model = _model_package["model"]
    prob = model.predict_proba(X)[0, 1]
    return float(prob)


# ── Risk Level ─────────────────────────────────────────────────────────────


def _ml_risk_level(probability: float) -> str:
    """Convert a cardiac risk probability to a plain-English risk tier."""
    for threshold, label in RISK_TIERS:
        if probability < threshold:
            return label
    return "High"


# ── Condition-Specific Scorers ────────────────────────────────────────────

# Each scorer takes the ML base score (0-40) and the user's data, then
# returns a dict with the final hybrid score and a list of boost explanations.


def _score_condition_hcm(data: UserHealthData, ml_base: float) -> dict:
    """
    Score HCM: ML base + family SCD + personal arrhythmia + MYH7 variant.

    Clinical rationale:
        HCM is autosomal dominant, so a family history of SCD is a
        significant red flag. Personal fainting/arrhythmia can indicate
        the outflow tract obstruction HCM causes. MYH7 is the most
        common HCM-associated gene, with high penetrance.
    """
    score = ml_base
    boosts: List[str] = []

    if data["family_scd"]:
        score += 20
        relation = data["family_scd_relation"] or "a relative"
        boosts.append(f"+20 Family history of sudden cardiac death ({relation}) — "
                      f"HCM is autosomal dominant")

    if data["personal_arrhythmia"]:
        score += 15
        boosts.append("+15 Personal fainting or irregular heartbeat — "
                      "can indicate HCM outflow obstruction")

    if data["variant_myh7"]:
        score += 35
        boosts.append("+35 MYH7 pathogenic variant — "
                      "direct genetic marker for HCM (high penetrance)")

    return {"score": min(score, 100), "ml_contribution": round(ml_base, 1), "boosts": boosts}


def _score_condition_lqts(data: UserHealthData, ml_base: float) -> dict:
    """
    Score LQTS: ML base + family SCD + personal arrhythmia + KCNQ1 variant.

    Clinical rationale:
        LQTS is a leading cause of sudden death in young people with
        no prior symptoms. Family SCD history carries heavy weight in
        the clinical Schwartz score. Unexplained syncope is one of the
        strongest Schwartz criteria. KCNQ1 mutations cause LQTS type 1,
        the most common form, triggered by exercise.
    """
    score = ml_base
    boosts: List[str] = []

    if data["family_scd"]:
        score += 20
        relation = data["family_scd_relation"] or "a relative"
        boosts.append(f"+20 Family history of sudden cardiac death ({relation}) — "
                      f"LQTS is a leading cause of sudden death in the young")

    if data["personal_arrhythmia"]:
        score += 20
        boosts.append("+20 Personal fainting or irregular heartbeat — "
                      "unexplained syncope is a strong Schwartz score criterion")

    if data["variant_kcnq1"]:
        score += 35
        boosts.append("+35 KCNQ1 pathogenic variant — "
                      "direct genetic marker for LQTS type 1 (most common form)")

    return {"score": min(score, 100), "ml_contribution": round(ml_base, 1), "boosts": boosts}


def _score_condition_fh(data: UserHealthData, ml_base: float) -> dict:
    """
    Score FH: ML base + family early MI + high LDL + LDLR variant.

    Clinical rationale:
        FH accelerates atherosclerosis from childhood, so a family
        history of early heart attack is a core Dutch Lipid Clinic
        Network criterion. LDL >= 190 mg/dL is the standard screening
        threshold for possible FH in untreated adults. LDLR mutations
        cause ~90% of confirmed FH cases.
    """
    score = ml_base
    boosts: List[str] = []

    if data["family_early_mi"]:
        score += 15
        boosts.append("+15 Family history of heart attack before age 50 — "
                      "FH accelerates atherosclerosis from a young age")

    if data["ldl_cholesterol"] >= 190:
        score += 25
        boosts.append(f"+25 LDL cholesterol of {data['ldl_cholesterol']} mg/dL — "
                      f"at or above the 190 mg/dL FH screening threshold")

    if data["variant_ldlr"]:
        score += 35
        boosts.append("+35 LDLR pathogenic variant — "
                      "direct genetic marker for FH (~90% of confirmed cases)")

    return {"score": min(score, 100), "ml_contribution": round(ml_base, 1), "boosts": boosts}


# Map condition code -> scorer function
_CONDITION_SCORERS = {
    "HCM": _score_condition_hcm,
    "LQTS": _score_condition_lqts,
    "FH": _score_condition_fh,
}

# ── Public API ─────────────────────────────────────────────────────────────


def predict_risk(data: UserHealthData) -> List[dict]:
    """
    Run the hybrid ML + rules predictor.

    Uses the best available ML model (NHANES XGBoost > Framingham Random
    Forest) and layers condition-specific rules-based boosts on top.

    Args:
        data: Parsed user health data from the intake form.

    Returns:
        List[dict]: Three condition assessments, sorted by hybrid_score
            descending. Each dict contains:
            - condition / full_name / genes  (identifiers)
            - ml_probability: raw cardiac probability from the ML model
            - ml_risk_level: plain-English tier ("Low", "Moderate", etc.)
            - ml_base_score: ML contribution to the score (0-40)
            - hybrid_score: combined ML + rules score (0-100)
            - boosts: list of plain-English boost explanations
            - model_source: which model was used ("nhanes" or "framingham")
    """
    # Get the ML cardiac probability (or fallback if model not yet trained)
    if model_available():
        model_source = get_model_source()
        cardiac_prob = _get_cardiac_probability(data)
    else:
        cardiac_prob = 0.10  # moderate default
        model_source = "none"

    ml_base = cardiac_prob * ML_MAX_SCORE     # 0-40
    risk_level = _ml_risk_level(cardiac_prob)

    # Build base dicts for each condition (shared ML fields)
    condition_defs = [
        ("HCM", "Hypertrophic Cardiomyopathy", ["MYH7", "MYBPC3"]),
        ("LQTS", "Long QT Syndrome", ["KCNQ1", "KCNH2", "SCN5A"]),
        ("FH", "Familial Hypercholesterolaemia", ["LDLR", "APOB"]),
    ]

    assessments = []
    for code, full_name, genes in condition_defs:
        entry = {
            "condition": code,
            "full_name": full_name,
            "genes": genes,
            "ml_probability": round(cardiac_prob, 3),
            "ml_risk_level": risk_level,
            "ml_base_score": round(ml_base, 1),
            "model_source": model_source,
        }

        # Apply condition-specific boosts
        scorer = _CONDITION_SCORERS[code]
        result = scorer(data, ml_base)
        entry["hybrid_score"] = result["score"]
        entry["boosts"] = result["boosts"]

        assessments.append(entry)

    # Sort descending by hybrid score so the riskiest condition is first
    assessments.sort(key=lambda a: a["hybrid_score"], reverse=True)
    return assessments
