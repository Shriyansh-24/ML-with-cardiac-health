"""
app.py — CardioGenome Flask entry point.

DESIGN NOTE: This file should stay "thin." Routes here only do three things:
    1. Receive the incoming request (form data, query params, etc.)
    2. Call out to a function in services/ to do the actual work
    3. Render a template with the result

Why keep it this way? If you put risk-scoring logic, API calls, or ML
inference directly inside a route function, this file balloons into an
unreadable mess as the app grows, and you can't test the logic without
spinning up a Flask server. Keeping routes thin means every route is a
short, obvious mapping from URL -> behavior -> page.

As we build out Modules 1-4, this file will grow by a route or two, but
each route's *body* should stay tiny — a few lines at most.
"""

from flask import Flask, render_template, request

from services import risk_profiler, predictor
from services.risk_profiler import FormParsingError

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """
    Render the input form (Module inputs described in the build spec).

    Returns:
        str: Rendered HTML for the landing page / intake form.
    """
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results() -> str:
    """
    Handle the submitted intake form and render the risk report.

    Wires up Module 1 + ML predictor (Step 4):
        1. Parse the raw form into a typed UserHealthData dict
        2. Run the hybrid ML + rules predictor for condition scores
        3. Run the rules-based condition profiler for explainable criteria
        4. Merge both into a combined view and render

    The merge works by condition code (HCM / LQTS / FH): the ML predictor
    provides the continuous score (0-100) and boosts, while the rules
    profiler provides the discrete criteria (out of 3) and plain-English
    reasons. Both contribute to the final report card.

    Returns:
        str: Rendered HTML for the results/report page. On malformed
            input, re-renders the intake form with an inline error.
    """
    try:
        user_data = risk_profiler.parse_form_data(request.form)
    except FormParsingError as error:
        return render_template("index.html", error_message=str(error)), 400

    # Get both ML and rules assessments
    ml_assessments = predictor.predict_risk(user_data)
    rules_assessments = risk_profiler.assess_risk(user_data)

    # Merge rules data into ML assessments by condition code
    rules_by_code = {a["condition"]: a for a in rules_assessments}
    for ml in ml_assessments:
        rules = rules_by_code.get(ml["condition"])
        if rules:
            ml["rules_score"] = rules["score"]
            ml["rules_max_score"] = rules["max_score"]
            ml["rules_reasons"] = rules["reasons"]

    global_chd = ml_assessments[0] if ml_assessments else None

    return render_template(
        "results.html",
        ml_assessments=ml_assessments,
        global_risk_level=global_chd["ml_risk_level"] if global_chd else "Unknown",
        global_probability=global_chd["ml_probability"] if global_chd else 0,
    )


if __name__ == "__main__":
    # debug=True gives auto-reload + interactive tracebacks while developing.
    # IMPORTANT: turn this off (or don't use it) before any real deployment —
    # the interactive debugger can execute arbitrary code if exposed publicly.
    app.run(debug=True)