# -*- coding: utf-8 -*-
"""Serverside callback for exporting the local research impact summary."""

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app_src import VERSION
from app_src.dialogs import save_file_dialog
from app_src.server import app
from app_src.usage_stats import (
    disable_usage_reporting,
    enable_usage_reporting,
    format_usage_summary,
    get_usage_report_url,
    read_usage_stats,
    sync_usage_reports,
)


@app.callback(
    Output("usage-summary-message", "children"),
    Input("usage-summary-button", "n_clicks"),
    prevent_initial_call=True,
)
def export_usage_summary(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    stats = read_usage_stats()
    summary = format_usage_summary(stats, app_version=VERSION)

    save_path = save_file_dialog("txt", "sleep_scoring_impact_summary.txt")
    if not save_path:
        return ""

    try:
        with open(save_path, "w", encoding="utf-8") as handle:
            handle.write(summary + "\n")
    except OSError as error:
        return f"Could not save the impact summary: {error}"

    return f"Saved impact summary to {save_path}."


@app.callback(
    Output("usage-reporting-confirm", "displayed"),
    Output("usage-reporting-confirm", "message"),
    Input("usage-reporting-button", "n_clicks"),
    prevent_initial_call=True,
)
def confirm_usage_reporting(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return (
        True,
        "Share anonymous usage totals for this copy of Sleep Scoring? "
        "The report contains a random app ID, completed-recording counts, "
        "scored hours, app version, and timestamps. It never contains file "
        "names, paths, signals, annotations, animal identifiers, or the "
        "local recording fingerprints used to prevent double-counting.",
    )


@app.callback(
    Output("usage-reporting-message", "children", allow_duplicate=True),
    Input("usage-reporting-confirm", "submit_n_clicks"),
    prevent_initial_call=True,
)
def enable_reporting(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    if not get_usage_report_url():
        return "Usage sharing is not available in this build."
    if not enable_usage_reporting():
        return "Could not enable usage sharing; this app folder may not be writable."

    status = sync_usage_reports()
    if status == "sent":
        return "Anonymous usage sharing is enabled and the current totals were sent."
    return "Anonymous usage sharing is enabled. Pending totals will retry when available."


@app.callback(
    Output("usage-reporting-message", "children", allow_duplicate=True),
    Input("usage-reporting-stop-button", "n_clicks"),
    prevent_initial_call=True,
)
def disable_reporting(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    if disable_usage_reporting():
        return "Anonymous usage sharing is off. Local usage totals are unchanged."
    return "Could not stop usage sharing; this app folder may not be writable."
