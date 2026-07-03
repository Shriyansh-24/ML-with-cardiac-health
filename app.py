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

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def index() -> str:
    """
    Render the input form (Module inputs described in the build spec).

    Returns:
        str: Rendered HTML for the landing page / intake form.

    NOTE: This is a placeholder for Step 1. In Step 2, this will render
    templates/index.html, which contains the full health-data form.
    """
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results() -> str:
    """
    Placeholder for the results route.

    This will eventually:
        1. Pull submitted form data (services/risk_profiler.py, Module 1)
        2. Run the ML model on that data (services/predictor.py)
        3. Fetch gene/variant context (services/clinvar_api.py, gwas_api.py)
        4. Load static gene-editing + equity data (data/*.json)
        5. Render templates/results.html with everything combined

    For now, it just confirms the route exists and returns a stub page,
    so we can verify the skeleton wires together correctly before adding
    any real logic.

    Returns:
        str: Rendered HTML for the results/report page.
    """
    return render_template("results.html")


if __name__ == "__main__":
    # debug=True gives auto-reload + interactive tracebacks while developing.
    # IMPORTANT: turn this off (or don't use it) before any real deployment —
    # the interactive debugger can execute arbitrary code if exposed publicly.
    app.run(debug=True)
