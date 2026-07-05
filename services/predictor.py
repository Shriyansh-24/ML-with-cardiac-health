"""
services/predictor.py — ML + rules hybrid predictor for CardioGenome.

WHAT THIS MODULE DOES
    Combines the Random Forest model trained on Framingham data with the
    rules-based risk factors that the model was never trained on (family
    history, genetic variants, personal symptoms).

    The model predicts 10-year CHD risk from 5 clinical features (age,
    sex, BP, cholesterol, heart rate). But each specific condition (HCM,
    LQTS, FH) has unique risk factors that the Framingham dataset never
    included — family history, genetic variant flags, symptom history.
    This module layers those on top of the ML prediction.

WHY HYBRID?
    A pure ML model would completely miss family history and genetic
    variants — those columns don't exist in Framingham. A pure rules
    system can't find subtle patterns in continuous data like age × BP
    interactions. Combining them gives us the best of both worlds.

HOW IT WORKS
    1. Load the trained model from ml/cardiac_model.pkl (lazy — only
       loads when the first request hits the results page)
    2. Map the user's form fields to the 5 model features
    3. Get the CHD probability from the model (0.0 - 1.0)
    4. Convert to a ML base score (0-40 scale)
    5. For each condition, add rules-based boosts (0-60 points) for
       family history, symptoms, and genetic variants
    6. Cap at 100 for a clean, interpretable hybrid score
"""

from typing import List, Optional
import os
import joblib
import pandas as pd

from services.risk_profiler import UserHealthData


# ── Constants ──────────────────────────────────────────────────────────────

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ml", "cardiac_model.pkl")
)

# How many points the ML model can contribute (out of 100)
ML_MAX_SCORE = 40

# Risk-level thresholds for the raw CHD probability
RISK_TIERS = [
    (0.05, "Low"),
    (0.10, "Low-Moderate"),
    (0.15, "Moderate"),
    (0.25, "Elevated"),
    (float("inf"), "High"),
]

# ── Model Loading (lazy) ──────────────────────────────────────────────────

_model_package = None  # module-level cache; loaded once on first request


def _ensure_model_loaded() -> None:
    """Load the saved model package from disk on first call."""
    global _model_package
    if _model_package is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Run `python ml/train_model.py` first to train and save the model."
        )
    _model_package = joblib.load(MODEL_PATH)


def model_available() -> bool:
    """Check if a trained model exists on disk (does not trigger a full load)."""
    return os.path.exists(MODEL_PATH)


# ── Feature Preparation ────────────────────────────────────────────────────


def _prepare_features(data: UserHealthData) -> pd.DataFrame:
    """
    Convert UserHealthData into the 5-column DataFrame the model expects.

    The model was trained on columns ['age', 'sex', 'trestbps', 'chol',
    'thalach'] in that exact order. This function maps form fields to
    those names and converts sex to binary.

    Args:
        data: Parsed user health data from the intake form.

    Returns:
        pd.DataFrame with exactly 1 row and the 5 model features.
    """
    # biological_sex: "male" -> 1, "female" -> 0
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


def _get_chd_probability(data: UserHealthData) -> float:
    """
    Run the trained Random Forest on the user's data.

    Args:
        data: Parsed user health data.

    Returns:
        float: Predicted probability of 10-year CHD (0.0 - 1.0).
    """
    _ensure_model_loaded()
    X = _prepare_features(data)
    prob = _model_package["model"].predict_proba(X)[0, 1]
    return float(prob)


# ── Risk Level ─────────────────────────────────────────────────────────────


def _ml_risk_level(probability: float) -> str:
    """Convert a CHD probability to a plain-English risk tier."""
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

    Produces a risk assessment for each of the three target conditions
    (HCM, LQTS, FH), combining the ML model's CHD probability with
    condition-specific rules-based boosts for factors the model never
    saw during training.

    Args:
        data: Parsed user health data from the intake form.

    Returns:
        List[dict]: Three condition assessments, sorted by hybrid_score
            descending. Each dict contains:
            - condition / full_name / genes  (identifiers)
            - ml_probability: raw CHD probability from the ML model
            - ml_risk_level: plain-English tier ("Low", "Moderate", etc.)
            - ml_base_score: ML contribution to the score (0-40)
            - hybrid_score: combined ML + rules score (0-100)
            - boosts: list of plain-English boost explanations
    """
    # Get the ML CHD probability (or fallback if model not yet trained)
    if model_available():
        chd_prob = _get_chd_probability(data)
    else:
        chd_prob = 0.10  # moderate default

    ml_base = chd_prob * ML_MAX_SCORE     # 0-40
    risk_level = _ml_risk_level(chd_prob)

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
            "ml_probability": round(chd_prob, 3),
            "ml_risk_level": risk_level,
            "ml_base_score": round(ml_base, 1),
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
