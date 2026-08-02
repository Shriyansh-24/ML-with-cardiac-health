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

from flask import Flask, render_template, request, redirect, url_for, flash

import json
import os

from services import risk_profiler, predictor, clinvar_api, gwas_api, equity, auth_service
from services.risk_profiler import FormParsingError

app = Flask(__name__)
# Required for Flask's signed session cookie (which holds Supabase tokens).
# In production this MUST be set via the SECRET_KEY env var — the fallback
# is only safe for local development.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-change-me")
# Send the session cookie only over HTTPS in production (set
# SESSION_COOKIE_SECURE=true on Render). Off by default so plain
# http://127.0.0.1 local development keeps working.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"


@app.context_processor
def inject_auth_context() -> dict:
    """
    Expose the logged-in user (and whether auth is configured) to every template.

    Making this a context processor means base.html can show the right nav
    (Log in / Sign up vs. My reports / Log out) on every page without each
    route having to pass the user along explicitly.
    """
    return {
        "current_user": auth_service.get_current_user(),
        "auth_configured": auth_service.is_configured(),
    }


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

    # Module 2: fetch real ClinVar variant data for MYH7 (HCM-associated gene)
    clinvar_data = clinvar_api.fetch_gene_variants("MYH7")

    # Step 6: fetch GWAS Catalog associations for LQTS and FH genes
    gwas_data = gwas_api.fetch_all_condition_associations()

    # Module 4: load equity dashboard data and generate Plotly charts
    equity_dashboard = equity.build_equity_dashboard()

    # Module 3: load static gene editing research dataset
    editing_path = os.path.join(os.path.dirname(__file__), "data", "gene_editing.json")
    if os.path.exists(editing_path):
        with open(editing_path, "r") as f:
            gene_editing_full = json.load(f)
        gene_editing_data = {
            "conditions": gene_editing_full.get("conditions", {}),
            "last_updated": gene_editing_full.get("last_updated", ""),
        }
    else:
        gene_editing_data = {"conditions": {}, "last_updated": ""}

    global_chd = ml_assessments[0] if ml_assessments else None
    model_source = global_chd["model_source"] if global_chd else "none"

    # Build a self-contained JSON snapshot of this report so a signed-in
    # user can save it via POST /save-report. The template embeds it in a
    # hidden form field; we re-serialize it on save. Keeping the payload
    # minimal (summary + conditions + form answers) means the saved copy
    # renders correctly even if live APIs go down later.
    report_payload = {
        "summary": {
            "probability": round(global_chd["ml_probability"], 4) if global_chd else 0,
            "risk_level": global_chd["ml_risk_level"] if global_chd else "Unknown",
            "model_source": model_source,
        },
        "conditions": [
            {
                "condition": item["condition"],
                "full_name": item["full_name"],
                "genes": item.get("genes", []),
                "ml_risk_level": item.get("ml_risk_level", "Unknown"),
                "hybrid_score": round(item.get("hybrid_score", 0), 1),
                "boosts": item.get("boosts", []),
                "rules_score": item.get("rules_score", 0),
                "rules_max_score": item.get("rules_max_score", 0),
                "rules_reasons": item.get("rules_reasons", []),
            }
            for item in ml_assessments
        ],
        "form_snapshot": dict(user_data),
    }

    return render_template(
        "results.html",
        ml_assessments=ml_assessments,
        global_risk_level=global_chd["ml_risk_level"] if global_chd else "Unknown",
        global_probability=global_chd["ml_probability"] if global_chd else 0,
        model_source=model_source,
        clinvar_data=clinvar_data,
        gwas_data=gwas_data,
        gene_editing=gene_editing_data,
        equity_dashboard=equity_dashboard,
        report_payload_json=json.dumps(report_payload),
    )


# ── Phase 2: accounts + saved reports ────────────────────────────────────


@app.route("/signup", methods=["GET", "POST"])
def signup() -> str:
    """
    Create an account (email + password) with an optional research opt-in.

    On success: if Supabase email confirmation is enabled (the default),
    the user is told to check their inbox; otherwise they're logged in
    immediately. Errors re-render the form with a friendly message.
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("password_confirm", "")
        research_opt_in = request.form.get("research_opt_in") == "yes"

        if not email or not password:
            return render_template(
                "signup.html",
                error_message="Please fill in both email and password.",
                email=email,
                research_opt_in=research_opt_in,
            )
        if password != confirm:
            return render_template(
                "signup.html",
                error_message="Passwords do not match.",
                email=email,
                research_opt_in=research_opt_in,
            )
        if len(password) < 6:
            return render_template(
                "signup.html",
                error_message="Password must be at least 6 characters.",
                email=email,
                research_opt_in=research_opt_in,
            )

        user, error = auth_service.signup(email, password, research_opt_in)
        if error:
            return render_template(
                "signup.html",
                error_message=error,
                email=email,
                research_opt_in=research_opt_in,
            )
        if user and user.get("pending_confirmation"):
            # Email confirmation is on — they need to click the emailed link
            # before they can log in.
            return render_template(
                "signup.html",
                info_message=(
                    "Account created! We've sent a confirmation link to your "
                    "email. Click it, then log in."
                ),
            )
        flash("Account created — welcome!", "success")
        return redirect(url_for("index"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    """
    Sign in with email + password.

    Supports an optional `next` query param (used when a signed-out user
    hits a protected page like /history) so they land where they were going.
    """
    next_url = request.args.get("next") or url_for("index")
    # Only allow local redirects — never let an open redirect follow a
    # user-supplied absolute URL.
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = url_for("index")

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        next_url = request.form.get("next") or next_url
        if not next_url.startswith("/") or next_url.startswith("//"):
            next_url = url_for("index")

        if not email or not password:
            return render_template("login.html", error_message="Please enter both email and password.", next=next_url)

        user, error = auth_service.login(email, password)
        if error:
            return render_template("login.html", error_message=error, email=email, next=next_url)
        flash(f"Welcome back, {user['email']}!", "success")
        return redirect(next_url)

    return render_template("login.html", next=next_url)


@app.route("/logout", methods=["POST"])
def logout() -> str:
    """
    Clear the session cookie (and Supabase tokens it holds).

    POST-only so a stray <img> or prefetch can't log the user out.
    """
    auth_service.logout()
    flash("You've been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/save-report", methods=["POST"])
def save_report() -> str:
    """
    Persist the submitted report JSON for the signed-in user.

    The results page embeds a JSON snapshot in a hidden form field; this
    route validates it and hands it to auth_service.save_report(), which
    also triggers the anonymised research insert when the user opted in.
    """
    if not auth_service.get_current_user():
        flash("Please log in to save a report.", "error")
        return redirect(url_for("login", next=url_for("index")))

    try:
        payload = json.loads(request.form.get("report_json", "{}"))
        summary = payload["summary"]
        conditions = payload["conditions"]
        form_snapshot = payload.get("form_snapshot", {})
    except (ValueError, KeyError, TypeError):
        flash("Sorry, that report couldn't be saved.", "error")
        return redirect(url_for("index"))

    report_id, error = auth_service.save_report(summary, conditions, form_snapshot)
    if error or not report_id:
        flash(error or "Sorry, that report couldn't be saved.", "error")
        return redirect(url_for("index"))
    flash("Report saved to your history.", "success")
    return redirect(url_for("report_detail", report_id=report_id))


@app.route("/history")
def history() -> str:
    """
    List the signed-in user's saved reports, newest first.

    If someone isn't logged in, send them to login with a `next` back here.
    """
    if not auth_service.get_current_user():
        return redirect(url_for("login", next=url_for("history")))
    reports = auth_service.list_reports()
    return render_template("history.html", reports=reports)


@app.route("/report/<report_id>")
def report_detail(report_id: str) -> str:
    """
    View one saved report.

    Row-level security means auth_service.get_report() only ever returns a
    row belonging to the signed-in user — so an id that belongs to someone
    else simply comes back as None and we show the not-found state.
    """
    if not auth_service.get_current_user():
        return redirect(url_for("login", next=url_for("report_detail", report_id=report_id)))
    report = auth_service.get_report(report_id)
    if not report:
        return render_template(
            "history.html",
            reports=auth_service.list_reports(),
            error_message="That report could not be found.",
        ), 404
    return render_template("report_detail.html", report=report)


@app.route("/delete-report/<report_id>", methods=["POST"])
def delete_report(report_id: str) -> str:
    """
    Delete one of the signed-in user's saved reports.

    POST-only, and RLS means only the owner's rows are ever deletable.
    """
    if not auth_service.get_current_user():
        return redirect(url_for("login", next=url_for("history")))
    auth_service.delete_report(report_id)
    flash("Report deleted.", "info")
    return redirect(url_for("history"))


if __name__ == "__main__":
    # Debug mode controlled by env var — off by default for production safety.
    # Set FLASK_DEBUG=1 to enable the interactive debugger in development.
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_enabled)