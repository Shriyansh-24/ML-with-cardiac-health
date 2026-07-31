"""
services/equity.py — Equity dashboard (Module 4).

Loads the static equity dataset from data/equity.json and renders it as
interactive Plotly charts that can be embedded in the results page.

Design:
    - All data is static JSON — no API calls, no caching needed.
    - Plotly charts are rendered to full HTML <div> strings via
      plotly.io.to_html() so the template just inserts them with
      {{ ... | safe }}.
    - The equity_data dict also carries the raw numbers and metadata
      for reference in the template even if JavaScript is off.
"""

import json
import os
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

# ── Chart styling constants ──────────────────────────────────────────
CHART_TEMPLATE = "plotly_white"
FONT_FAMILY = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

pio.templates.default = CHART_TEMPLATE


def _load_data() -> dict[str, Any]:
    """Load the equity JSON from the data directory.

    Returns:
        dict: The full equity dataset, or an empty dict on error.
    """
    path = os.path.join(os.path.dirname(__file__), "..", "data", "equity.json")
    path = os.path.normpath(path)
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"charts": {}, "_metadata": {"error": "Equity data file not found."}}


def _simple_bar_chart(
    chart_def: dict[str, Any],
    height: int = 300,
) -> str:
    """Build a simple vertical bar chart from {label, value, marker_color} items.

    The title/subtitle are NOT baked into the Plotly figure — they are shown
    by the results template in a card header instead, so we keep them out of
    the SVG and avoid duplicate titles.

    Args:
        chart_def: Chart definition from the JSON (title, subtitle, data list).
        height: Chart height in pixels.

    Returns:
        str: Full Plotly HTML div string.
    """
    data = chart_def.get("data", [])

    fig = go.Figure(
        data=[
            go.Bar(
                x=[d["label"] for d in data],
                y=[d["value"] for d in data],
                marker_color=[d.get("marker_color", "#3498db") for d in data],
                text=[f"{d['value']}%" for d in data],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>%{yaxis.title.text}: %{y}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        yaxis={
            "title": chart_def.get("yaxis_label", "%"),
            "range": [0, 100],
            "dtick": 20,
            "gridcolor": "#f0f0f0",
        },
        xaxis={
            "categoryorder": "array",
            "categoryarray": [d["label"] for d in data],
        },
        height=height,
        margin={"t": 12, "b": 40, "l": 44, "r": 12},
        font={"family": FONT_FAMILY},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _grouped_bar_chart(
    chart_def: dict[str, Any],
    height: int = 300,
) -> str:
    """Build a grouped bar chart comparing two series (e.g. trial vs population).

    Args:
        chart_def: Chart definition with data.groups (list) and
            data.series_labels dict mapping series keys to display names.
        height: Chart height in pixels.

    Returns:
        str: Full Plotly HTML div string.
    """
    groups = chart_def.get("data", {}).get("groups", [])
    series = chart_def.get("data", {}).get("series_labels", {})
    labels = [g["label"] for g in groups]

    fig = go.Figure()

    for series_key, series_label in series.items():
        fig.add_trace(
            go.Bar(
                name=series_label,
                x=labels,
                y=[g.get(series_key, 0) for g in groups],
                marker_color=[g.get("marker_color", "#3498db") for g in groups]
                if series_key == list(series.keys())[0]
                else "#bbb",
                text=[f"{g.get(series_key, 0)}%" for g in groups],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=f"<b>%{{x}}</b><br>{series_label}: %{{y}}%<extra></extra>",
            )
        )

    fig.update_layout(
        yaxis={
            "title": chart_def.get("yaxis_label", "%"),
            "range": [0, 85],
            "dtick": 20,
            "gridcolor": "#f0f0f0",
        },
        barmode="group",
        height=height,
        # The horizontal legend sits above the plot (y: 1.02), so we keep a
        # taller top margin here than in _simple_bar_chart to avoid clipping.
        margin={"t": 40, "b": 40, "l": 44, "r": 12},
        font={"family": FONT_FAMILY},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def _compute_highlights(charts_defs: dict[str, Any]) -> list[dict[str, str]]:
    """Derive 4 headline 'takeaway' stats from the raw chart data.

    These power the summary cards at the top of the equity tab, so a visitor
    gets the punchline before scrolling into the full charts.

    Args:
        charts_defs: The raw charts dict from data/equity.json.

    Returns:
        list of dicts: [{label, value, note}, ...]
    """
    highlights: list[dict[str, str]] = []

    # ── Racial gap in testing access ──
    race = charts_defs.get("testing_by_race", {}).get("data", [])
    if race:
        best = max(race, key=lambda d: d["value"])
        worst = min(race, key=lambda d: d["value"])
        ratio = best["value"] / worst["value"] if worst["value"] else 0
        highlights.append({
            "label": "Race gap in testing",
            "value": f"{best['value']}% vs {worst['value']}%",
            "note": (
                f"{best['label']} patients are {ratio:.1f}× more likely to receive "
                f"genetic testing than {worst['label']} patients."
            ),
        })

    # ── Income divide in access ──
    income = charts_defs.get("testing_by_income", {}).get("data", [])
    if income:
        best = max(income, key=lambda d: d["value"])
        worst = min(income, key=lambda d: d["value"])
        gap = best["value"] - worst["value"]
        highlights.append({
            "label": "Income divide",
            "value": f"{gap} pt gap",
            "note": (
                f"Testing access ranges from {worst['value']}% ({worst['label']}) "
                f"to {best['value']}% ({best['label']})."
            ),
        })

    # ── Trial representation imbalance ──
    groups = charts_defs.get("trial_representation", {}).get("data", {}).get("groups", [])
    if groups:
        over = max(groups, key=lambda g: (g.get("trial_pct", 0) - g.get("population_pct", 0)))
        under = min(groups, key=lambda g: (g.get("trial_pct", 0) - g.get("population_pct", 0)))
        over_delta = over.get("trial_pct", 0) - over.get("population_pct", 0)
        under_delta = under.get("population_pct", 0) - under.get("trial_pct", 0)
        highlights.append({
            "label": "Trial representation",
            "value": f"+{over_delta}pt / -{under_delta}pt",
            "note": (
                f"{over['label']} patients make up {over['trial_pct']}% of trial "
                f"participants vs {over['population_pct']}% of the U.S. population — "
                f"while {under['label']} patients are the most under-represented."
            ),
        })

    # ── Global access gap ──
    glob = charts_defs.get("global_access", {}).get("data", [])
    if glob:
        best = max(glob, key=lambda d: d["value"])
        worst = min(glob, key=lambda d: d["value"])
        ratio = best["value"] / worst["value"] if worst["value"] else 0
        highlights.append({
            "label": "Global access gap",
            "value": f"{ratio:.1f}×",
            "note": (
                f"Eligible patients in {best['label']} are about {ratio:.1f}× more "
                f"likely to access gene therapies than those in {worst['label']}."
            ),
        })

    return highlights


def build_equity_dashboard() -> dict[str, Any]:
    """Build the full equity dashboard payload for the results template.

    Returns:
        dict with keys:
            - charts: dict of {chart_key: {"html": str, "title": str, ...}}
            - metadata: dict with source info and error info
            - has_data: bool
    """
    data = _load_data()
    charts_defs = data.get("charts", {})
    metadata = data.get("_metadata", {})

    if not charts_defs:
        return {
            "charts": {},
            "metadata": metadata,
            "has_data": False,
        }

    output_charts: dict[str, Any] = {}
    builders = {
        "bar": _simple_bar_chart,
        "grouped_bar": _grouped_bar_chart,
    }

    for chart_key, chart_def in charts_defs.items():
        chart_type = chart_def.get("type", "bar")
        builder = builders.get(chart_type)
        if builder is None:
            continue
        try:
            chart_html = builder(chart_def)
            output_charts[chart_key] = {
                "html": chart_html,
                "title": chart_def.get("title", ""),
                "subtitle": chart_def.get("subtitle", ""),
                "source_note": chart_def.get("source_note", ""),
            }
        except Exception as exc:
            output_charts[chart_key] = {
                "html": "",
                "title": chart_def.get("title", ""),
                "subtitle": chart_def.get("subtitle", ""),
                "source_note": "",
                "error": str(exc),
            }

    return {
        "charts": output_charts,
        "highlights": _compute_highlights(charts_defs),
        "metadata": metadata,
        "has_data": bool(output_charts),
    }
