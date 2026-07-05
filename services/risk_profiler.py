"""
services/risk_profiler.py — Module 1: rules-based cardiac risk profiler.

WHAT THIS MODULE DOES
    Takes the user's self-reported personal/family health data and produces
    a ranked list of the three target conditions (HCM, LQTS, FH), each with
    a transparency score and plain-English reasons for that score.

WHAT THIS MODULE IS NOT
    It is NOT a diagnostic tool. Real genetic risk assessment involves
    clinician judgment, ECGs, echocardiograms, and formal scoring systems
    (e.g. the Schwartz score for LQTS, the Dutch Lipid Clinic Network
    criteria for FH). This module uses deliberately simplified, transparent
    proxies for those criteria — see the comments on each `assess_*`
    function for exactly what's being approximated and what's left out.

DESIGN NOTE — why a scoring function per condition instead of one big
if/else block: each condition has genuinely different risk factors and
genes. Separating them means you can test, tweak, or explain HCM logic
without touching LQTS or FH logic, and it mirrors how a clinician would
actually reason — one condition at a time, not one giant decision tree.
"""

from typing import Mapping, Optional, TypedDict, List


class UserHealthData(TypedDict):
    """Clean, typed representation of one user's submitted form data."""
    biological_sex: str            # "male" | "female"
    age: int
    ethnicity: str                 # one of the dropdown options
    family_scd: bool                # family history of sudden cardiac death
    family_scd_relation: Optional[str]  # "parent" | "sibling" | "grandparent" | None
    family_early_mi: bool           # family history of heart attack before age 50
    personal_arrhythmia: bool       # personal history of fainting / irregular heartbeat
    resting_hr: int                 # beats per minute
    systolic_bp: int                # mmHg
    total_cholesterol: int          # mg/dL
    ldl_cholesterol: int            # mg/dL
    variant_myh7: bool              # self-reported MYH7 pathogenic variant (HCM)
    variant_kcnq1: bool             # self-reported KCNQ1 pathogenic variant (LQTS)
    variant_ldlr: bool              # self-reported LDLR pathogenic variant (FH)


class ConditionAssessment(TypedDict):
    """Result of scoring one condition against the user's data."""
    condition: str          # short code, e.g. "HCM"
    full_name: str          # e.g. "Hypertrophic Cardiomyopathy"
    genes: List[str]        # genes associated with this condition
    score: int              # how many risk criteria were met
    max_score: int          # total possible criteria, for normalising later
    reasons: List[str]      # plain-English explanation for each point scored


class FormParsingError(ValueError):
    """Raised when submitted form data is missing or malformed.

    Kept as its own exception (rather than letting a bare ValueError or
    KeyError bubble up) so app.py can catch this specific case and show
    the user a friendly "please fill out the form correctly" message,
    while still letting genuinely unexpected bugs surface as real errors.
    """
    pass


def _parse_bool(value: Optional[str]) -> bool:
    """
    Convert an HTML form value into a bool.

    HTML forms only send a field at all if a checkbox is checked, and
    yes/no radio buttons arrive as the literal strings "yes" or "no" —
    there's no native boolean type. This centralises that conversion so
    we don't repeat the same "yes"/"on" string-matching everywhere.

    Args:
        value: The raw form value, or None if the field wasn't submitted.

    Returns:
        bool: True if the value represents an affirmative answer.
    """
    if value is None:
        return False
    return value.strip().lower() in {"yes", "on", "true", "1"}


def parse_form_data(form: Mapping[str, str]) -> UserHealthData:
    """
    Convert raw submitted form data into a clean, typed UserHealthData dict.

    Accepts any dict-like object with a `.get()` method — this includes
    both a plain Python dict (handy for unit testing) and Flask's
    `request.form` (an ImmutableMultiDict), so this function has no
    dependency on Flask itself.

    Args:
        form: Raw form data, e.g. Flask's `request.form`.

    Returns:
        UserHealthData: Parsed, typed health data ready for scoring.

    Raises:
        FormParsingError: If a required numeric field is missing or isn't
            a valid number. Checkbox/optional fields never raise — they
            just default to False/None if absent.
    """
    def require_int(field_name: str) -> int:
        raw = form.get(field_name)
        if raw is None or raw.strip() == "":
            raise FormParsingError(f"Missing required field: {field_name}")
        try:
            return int(raw)
        except ValueError:
            raise FormParsingError(f"'{field_name}' must be a whole number, got: {raw!r}")

    def require_str(field_name: str) -> str:
        raw = form.get(field_name)
        if raw is None or raw.strip() == "":
            raise FormParsingError(f"Missing required field: {field_name}")
        return raw.strip()

    family_scd = _parse_bool(form.get("family_scd"))

    return UserHealthData(
        biological_sex=require_str("biological_sex"),
        age=require_int("age"),
        ethnicity=require_str("ethnicity"),
        family_scd=family_scd,
        # Only meaningful if family_scd is True; otherwise the form field
        # may be blank/absent, which is fine — we just store None.
        family_scd_relation=form.get("family_scd_relation") or None,
        family_early_mi=_parse_bool(form.get("family_early_mi")),
        personal_arrhythmia=_parse_bool(form.get("personal_arrhythmia")),
        resting_hr=require_int("resting_hr"),
        systolic_bp=require_int("systolic_bp"),
        total_cholesterol=require_int("total_cholesterol"),
        ldl_cholesterol=require_int("ldl_cholesterol"),
        # Variant checkboxes are always optional — the spec says the model
        # still runs if the user has no genetic data, so these default to
        # False rather than raising if absent.
        variant_myh7=_parse_bool(form.get("variant_myh7")),
        variant_kcnq1=_parse_bool(form.get("variant_kcnq1")),
        variant_ldlr=_parse_bool(form.get("variant_ldlr")),
    )


def assess_hcm_risk(data: UserHealthData) -> ConditionAssessment:
    """
    Score Hypertrophic Cardiomyopathy (HCM) risk indicators.

    CLINICAL APPROXIMATION NOTE:
    Real HCM risk assessment looks at family history of sudden cardiac
    death, unexplained syncope, and structural findings on echocardiogram
    (e.g. septal wall thickness) — none of which we can measure from a
    home form. We proxy structural/electrical warning signs with the
    "personal history of fainting or irregular heartbeat" question, and
    genetic confirmation with the self-reported MYH7 flag. This is a
    coarse screening heuristic, not a diagnostic criterion set.

    Args:
        data: The user's parsed health data.

    Returns:
        ConditionAssessment: HCM score, out of a max of 3, with reasons.
    """
    score = 0
    reasons: List[str] = []

    # Criterion 1: family history of sudden cardiac death (any relation).
    # HCM is autosomal dominant, so a first- or second-degree relative
    # having sudden cardiac death is a recognised red flag.
    if data["family_scd"]:
        score += 1
        relation = data["family_scd_relation"] or "a relative"
        reasons.append(f"Family history of sudden cardiac death ({relation})")

    # Criterion 2: personal fainting/irregular heartbeat, a rough proxy for
    # the outflow obstruction or arrhythmia symptoms HCM can cause.
    if data["personal_arrhythmia"]:
        score += 1
        reasons.append("Personal history of fainting or irregular heartbeat")

    # Criterion 3: self-reported MYH7 pathogenic variant — the strongest
    # signal here, since it's a direct genetic finding rather than a
    # symptom-based proxy.
    if data["variant_myh7"]:
        score += 1
        reasons.append("Self-reported MYH7 pathogenic variant")

    return ConditionAssessment(
        condition="HCM",
        full_name="Hypertrophic Cardiomyopathy",
        genes=["MYH7", "MYBPC3"],
        score=score,
        max_score=3,
        reasons=reasons,
    )


def assess_lqts_risk(data: UserHealthData) -> ConditionAssessment:
    """
    Score Long QT Syndrome (LQTS) risk indicators.

    CLINICAL APPROXIMATION NOTE:
    The real Schwartz score for LQTS combines ECG QTc duration, syncope
    history (especially with exercise/stress/startle), and family history
    of confirmed LQTS or sudden death under age 30. We can't measure QTc
    from a form, so we proxy with family SCD history and personal
    fainting/arrhythmia, plus the KCNQ1 genetic flag if available.

    Args:
        data: The user's parsed health data.

    Returns:
        ConditionAssessment: LQTS score, out of a max of 3, with reasons.
    """
    score = 0
    reasons: List[str] = []

    # Criterion 1: family history of sudden cardiac death — LQTS is a
    # leading cause of sudden death in otherwise healthy young people,
    # so this history carries real weight in the Schwartz score too.
    if data["family_scd"]:
        score += 1
        relation = data["family_scd_relation"] or "a relative"
        reasons.append(f"Family history of sudden cardiac death ({relation})")

    # Criterion 2: personal fainting/irregular heartbeat — syncope,
    # especially unexplained, is one of the strongest Schwartz score items.
    if data["personal_arrhythmia"]:
        score += 1
        reasons.append("Personal history of fainting or irregular heartbeat")

    # Criterion 3: self-reported KCNQ1 pathogenic variant (LQTS type 1,
    # the most common form).
    if data["variant_kcnq1"]:
        score += 1
        reasons.append("Self-reported KCNQ1 pathogenic variant")

    return ConditionAssessment(
        condition="LQTS",
        full_name="Long QT Syndrome",
        genes=["KCNQ1", "KCNH2", "SCN5A"],
        score=score,
        max_score=3,
        reasons=reasons,
    )


def assess_fh_risk(data: UserHealthData) -> ConditionAssessment:
    """
    Score Familial Hypercholesterolaemia (FH) risk indicators.

    CLINICAL APPROXIMATION NOTE:
    The real Dutch Lipid Clinic Network criteria assign points for LDL
    level bands, family history of premature coronary disease, physical
    signs (e.g. tendon xanthomas), and genetic testing — then classify
    "definite/probable/possible/unlikely" FH from a point total. We proxy
    this with a simplified LDL cutoff, family history of early heart
    attack, and the LDLR genetic flag. Real LDL cutoffs are age- and
    treatment-dependent; 190 mg/dL is a commonly cited screening
    threshold for untreated adults, used here as a simple proxy.

    Args:
        data: The user's parsed health data.

    Returns:
        ConditionAssessment: FH score, out of a max of 3, with reasons.
    """
    score = 0
    reasons: List[str] = []

    # Criterion 1: elevated LDL cholesterol. 190 mg/dL is a commonly used
    # untreated-adult screening threshold for possible FH — a real
    # clinical tool would adjust this by age and treatment status.
    LDL_SCREENING_THRESHOLD_MG_DL = 190
    if data["ldl_cholesterol"] >= LDL_SCREENING_THRESHOLD_MG_DL:
        score += 1
        reasons.append(
            f"LDL cholesterol of {data['ldl_cholesterol']} mg/dL is at or above "
            f"the {LDL_SCREENING_THRESHOLD_MG_DL} mg/dL screening threshold"
        )

    # Criterion 2: family history of early heart attack (before age 50) —
    # a core Dutch Lipid Clinic Network criterion, since FH accelerates
    # atherosclerosis from a young age.
    if data["family_early_mi"]:
        score += 1
        reasons.append("Family history of heart attack before age 50")

    # Criterion 3: self-reported LDLR pathogenic variant — LDLR mutations
    # cause the large majority of confirmed FH cases.
    if data["variant_ldlr"]:
        score += 1
        reasons.append("Self-reported LDLR pathogenic variant")

    return ConditionAssessment(
        condition="FH",
        full_name="Familial Hypercholesterolaemia",
        genes=["LDLR", "APOB"],
        score=score,
        max_score=3,
        reasons=reasons,
    )


def assess_risk(data: UserHealthData) -> List[ConditionAssessment]:
    """
    Run all three condition-specific scorers and rank the results.

    Args:
        data: The user's parsed health data.

    Returns:
        List[ConditionAssessment]: All three conditions, sorted from
            highest to lowest score. Ties keep a fixed HCM/LQTS/FH order
            (Python's sort is stable) rather than being arbitrary.
    """
    assessments = [
        assess_hcm_risk(data),
        assess_lqts_risk(data),
        assess_fh_risk(data),
    ]
    # Sort descending by score. `reverse=True` combined with Python's
    # stable sort means ties keep their original (HCM, LQTS, FH) order.
    assessments.sort(key=lambda a: a["score"], reverse=True)
    return assessments