# -*- coding: utf-8 -*-
"""Serverside callback for exporting the local research impact summary."""

from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from app_src import VERSION
from app_src.dialogs import save_file_dialog
from app_src.server import app
from app_src.usage_stats import format_usage_summary, read_usage_stats


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
