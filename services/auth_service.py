"""
services/auth_service.py — Phase 2: Supabase auth + report persistence.

WHAT THIS MODULE DOES
    Wraps the Supabase Python client so the Flask layer never touches
    Supabase directly. Handles:
        - signup / login / logout (email + password)
        - token storage in Flask's signed session cookie
        - saving / listing / viewing / deleting the current user's reports
        - anonymous research-data contribution (opt-in only, service-role key)

DESIGN NOTES
    - Tokens live in Flask's signed session cookie (secret on the server),
      never in browser localStorage — safer against XSS.
    - Per-user data is protected by Postgres Row Level Security: every
      query goes out with the user's JWT, so Supabase only returns/inserts
      rows where user_id = auth.uid(). The app never trusts the client to
      say "this is mine" — the database enforces it.
    - The service-role key is used ONLY for the anonymised research_data
      insert (a table with no user_id column, by design). It never touches
      user data, and it never leaves the server.
    - Graceful degradation: if SUPABASE_URL / SUPABASE_ANON_KEY are missing
      (local dev before Phase 2 config), every function behaves as "no auth
      configured" instead of crashing — the app keeps working exactly like
      Phase 1.
"""

import os

from flask import g, session
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

# Keys used in Flask's session cookie
SESSION_ACCESS = "sb_access_token"
SESSION_REFRESH = "sb_refresh_token"


def is_configured() -> bool:
    """True when the Supabase env vars are present (accounts are live)."""
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def _client(service_role: bool = False):
    """Create a Supabase client with the anon or service-role key."""
    key = SUPABASE_SERVICE_ROLE_KEY if service_role else SUPABASE_ANON_KEY
    return create_client(SUPABASE_URL, key)


def _friendly_error(error: Exception) -> str:
    """Map raw Supabase/gotrue exceptions to human-readable messages."""
    msg = str(error)
    low = msg.lower()
    if "invalid login credentials" in low:
        return "Incorrect email or password."
    if "email not confirmed" in low:
        return "Please confirm your email address first (check your inbox)."
    if "already registered" in low or "already exists" in low:
        return "An account with this email already exists."
    if "password should be at least" in low or "at least 6" in low:
        return "Password must be at least 6 characters."
    if "rate limit" in low:
        return "Too many attempts — please wait a moment and try again."
    return f"Something went wrong: {msg[:200]}"


def _store_session(auth_session) -> None:
    """Persist Supabase tokens into the Flask session cookie."""
    session[SESSION_ACCESS] = auth_session.access_token
    session[SESSION_REFRESH] = auth_session.refresh_token


def signup(email: str, password: str, research_opt_in: bool):
    """
    Create a new account.

    Args:
        email: User's email address.
        password: Plain-text password (min 6 chars).
        research_opt_in: Whether the user consented to anonymous research.

    Returns:
        (user_dict, error_message). user_dict is None on failure; if
        "pending_confirmation" is True the user must click an email link
        before they can log in (Supabase's default email confirmation).
    """
    if not is_configured():
        return None, "Accounts are not enabled yet."
    try:
        res = _client().auth.sign_up({
            "email": email,
            "password": password,
            "options": {"data": {"research_opt_in": bool(research_opt_in)}},
        })
    except Exception as e:
        return None, _friendly_error(e)

    user = res.user
    if user is None:
        return None, "Could not create account. Please try again."

    # Email confirmation is on by default in Supabase: if no session came
    # back, the user must click the link we emailed before logging in.
    if res.session is None:
        return {
            "id": user.id,
            "email": user.email,
            "research_opt_in": bool(research_opt_in),
            "pending_confirmation": True,
        }, None

    _store_session(res.session)
    return get_current_user(), None


def login(email: str, password: str):
    """
    Sign an existing user in and store tokens in the Flask session.

    Returns:
        (user_dict, error_message).
    """
    if not is_configured():
        return None, "Accounts are not enabled yet."
    try:
        res = _client().auth.sign_in_with_password({
            "email": email,
            "password": password,
        })
    except Exception as e:
        return None, _friendly_error(e)

    if res.session is None:
        return None, "Could not sign in. Please try again."

    _store_session(res.session)
    return get_current_user(), None


def logout() -> None:
    """Clear Supabase tokens from the Flask session cookie."""
    session.pop(SESSION_ACCESS, None)
    session.pop(SESSION_REFRESH, None)


def get_current_user():
    """
    Return the logged-in user dict, or None.

    Validates the stored access token against Supabase (refreshing it once
    if expired) so stale sessions don't linger. When not logged in, or when
    Supabase isn't configured, returns None without any network call.
    """
    # Cache per-request: routes + the context processor call this multiple
    # times per request, and each call would otherwise be a Supabase API
    # round-trip (plus a token refresh attempt on expiry).
    if hasattr(g, "current_user"):
        return g.current_user

    if not is_configured():
        g.current_user = None
        return None
    access = session.get(SESSION_ACCESS)
    if not access:
        g.current_user = None
        return None

    client = _client()
    try:
        client.auth.set_session(access, session.get(SESSION_REFRESH, ""))
        res = client.auth.get_user()
    except Exception:
        # Access token likely expired — try one refresh with the stored
        # refresh token before giving up.
        try:
            res = client.auth.refresh_session(session.get(SESSION_REFRESH, ""))
            if res.session:
                _store_session(res.session)
        except Exception:
            logout()
            g.current_user = None
            return None

    user = res.user
    if user is None:
        logout()
        g.current_user = None
        return None

    meta = user.user_metadata or {}
    result = {
        "id": user.id,
        "email": user.email,
        "research_opt_in": bool(meta.get("research_opt_in", False)),
    }
    g.current_user = result
    return result


def _authed_client():
    """
    Build a client carrying the current user's JWT (for RLS-scoped queries).

    Returns None when not logged in or Supabase isn't configured.
    """
    if not is_configured() or not session.get(SESSION_ACCESS):
        return None
    client = _client()
    client.auth.set_session(session[SESSION_ACCESS], session.get(SESSION_REFRESH, ""))
    return client


def save_report(summary: dict, conditions: list, form_snapshot: dict):
    """
    Persist a report for the current user.

    Returns:
        (report_id, error_message). report_id is None on failure.
    """
    user = get_current_user()
    if user is None:
        return None, "Please sign in to save your report."
    client = _authed_client()
    if client is None:
        return None, "Please sign in to save your report."

    try:
        res = client.table("reports").insert({
            "user_id": user["id"],
            "summary": summary,
            "conditions": conditions,
            "form_snapshot": form_snapshot,
        }).execute()
        rows = res.data
    except Exception as e:
        return None, _friendly_error(e)

    if not rows:
        return None, "Could not save the report. Please try again."

    # Best-effort anonymised research contribution (opt-in only). Never
    # fails the save if the research insert errors.
    if user.get("research_opt_in"):
        _contribute_research(summary, conditions)

    return rows[0].get("id"), None


def list_reports() -> list:
    """Return the current user's reports (id, created_at, summary), newest first."""
    client = _authed_client()
    if client is None:
        return []
    try:
        res = (
            client.table("reports")
            .select("id, created_at, summary")
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def get_report(report_id: str):
    """Return one report row, or None (RLS means only the owner can fetch it)."""
    client = _authed_client()
    if client is None:
        return None
    try:
        res = (
            client.table("reports")
            .select("*")
            .eq("id", report_id)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def delete_report(report_id: str) -> bool:
    """Delete one of the current user's reports. Returns True on success."""
    client = _authed_client()
    if client is None:
        return False
    try:
        client.table("reports").delete().eq("id", report_id).execute()
        return True
    except Exception:
        return False


def _contribute_research(summary: dict, conditions: list) -> None:
    """
    Insert an ANONYMOUS copy of a report into research_data.

    No user_id is stored — the payload is de-identified by design. Uses the
    service-role key (server-only) because the table's RLS grants access to
    nobody but the service role. Failures are swallowed: research collection
    should never break the user-facing save flow.
    """
    if not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        _client(service_role=True).table("research_data").insert({
            "payload": {"summary": summary, "conditions": conditions},
        }).execute()
    except Exception:
        pass
