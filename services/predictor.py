"""
services/predictor.py — ML + rules hybrid predictor for CardioGenome.

WHAT THIS MODULE DOES
    Runs the NHANES XGBoost model (ml/cardiac_model_nhanes.pkl) to get a
    cardiac risk probability, then layers condition-specific rules-based
    boosts on top for factors the ML model was never trained on (family
    history, genetic variants, personal symptoms).

MODEL: NHANES XGBoost
    - Trained on CDC NHANES 2017-2023 (~16,800 samples, 7 features)
    - Target: composite CVD (CHF, CHD, MI, stroke, angina)
    - ROC-AUC: 0.818 (tuned), 0.761 ± 0.018 (10-fold CV)
    - Uses Age, Systolic_BP, Total_Cholesterol from form +
      BMI, waist circumference, diastolic BP, CRP (imputed with medians)

HOW THE HYBRID WORKS
    The ML model predicts general cardiac risk from clinical data. But
    each condition (HCM, LQTS, FH) has unique risk factors that the
    NHANES dataset never included — family history of SCD, personal
    arrhythmia, genetic variant flags. This module layers those on top
    of the ML prediction.

    ML probability (0-1) × 40 = ML base score (0-40)
    + Rules boosts (family history, symptoms, variants) = 0-60
    = Hybrid score (0-100), capped at 100
"""

from typing import List, Dict
import os
import joblib
import pandas as pd

from services.risk_profiler import UserHealthData


# ── Model Path ─────────────────────────────────────────────────────────────

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "cardiac_model_nhanes.pkl")
)

# How many points the ML model can contribute (out of 100)
ML_MAX_SCORE = 40

# Risk-level thresholds for the raw cardiac probability
RISK_TIERS = [
    (0.06, "Low"),
    (0.12, "Low-Moderate"),
    (0.18, "Moderate"),
    (0.28, "Elevated"),
    (float("inf"), "High"),
]

# ── Model Loading (lazy, single model) ────────────────────────────────────

_model_package = None  # module-level cache; loaded once on first request


def _ensure_model_loaded() -> None:
    """Load the NHANES XGBoost model from disk on first call."""
    global _model_package
    if _model_package is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"NHANES model not found at {MODEL_PATH}. "
            f"Run `python ml/train_nhanes.py` first to train and save the model."
        )
    _model_package = joblib.load(MODEL_PATH)


def model_available() -> bool:
    """Check if the NHANES model exists on disk."""
    return os.path.exists(MODEL_PATH)


def get_model_source() -> str:
    """Return which model is available ("nhanes" or "none")."""
    return "nhanes" if model_available() else "none"


# ── Feature Preparation ────────────────────────────────────────────────────


def _prepare_features(data: UserHealthData) -> pd.DataFrame:
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

    # Calculate BMI from height/weight if both are provided (optional fields)
    height_cm = data.get("height_cm")
    weight_kg = data.get("weight_kg")
    if height_cm is not None and weight_kg is not None and height_cm > 0:
        # BMI = weight (kg) / (height in meters)^2
        height_m = height_cm / 100.0
        features["BMI"] = round(weight_kg / (height_m * height_m), 1)

    # Impute any remaining missing features with training-set medians
    for feat_name in feature_names:
        if feat_name not in features:
            features[feat_name] = float(impute_medians.get(feat_name, 0.0))

    # Ensure features are in the exact order the model expects
    ordered = {fn: features[fn] for fn in feature_names}
    return pd.DataFrame([ordered])


# ── ML Prediction ──────────────────────────────────────────────────────────


def _get_cardiac_probability(data: UserHealthData) -> float:
    """
    Run the NHANES XGBoost model on the user's data.

    Args:
        data: Parsed user health data.

    Returns:
        float: Predicted cardiac risk probability (0.0 - 1.0).
    """
    _ensure_model_loaded()
    X = _prepare_features(data)
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

    Uses the NHANES XGBoost model and layers condition-specific
    rules-based boosts on top.

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
        cardiac_prob = _get_cardiac_probability(data)
        model_source = "nhanes"
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
