# -*- coding: utf-8 -*-
"""Serverside callbacks for annotation history, undo, and saving results."""

import shutil
import threading

import dash
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from scipy.io import loadmat, savemat

from app_src.dialogs import save_file_dialog
from app_src.postprocessing import (
    get_first_unscored_segment,
    get_pred_label_stats,
    get_sleep_segments,
    standardize,
)
from app_src.server import TEMP_PATH, app, cache
from app_src.sleep_score_layers import normalize_sleep_scores
from app_src.usage_stats import record_scored_recording, sync_usage_reports


@app.callback(
    Output("undo-button", "style", allow_duplicate=True),
    Input("updated-sleep-scores-store", "data"),
    State("user-sleep-scores-store", "data"),
    prevent_initial_call=True,
)
def update_sleep_scores_history(updated_sleep_scores, user_sleep_scores):
    """Store the displayed score state and its sparse user-label layer together."""
    undo_button_style = {"visibility": "hidden"}
    if not updated_sleep_scores:
        return undo_button_style

    sleep_scores_history = cache.get("sleep_scores_history")
    updated_sleep_scores = np.array(updated_sleep_scores, dtype=float)
    user_sleep_scores_history = cache.get("user_sleep_scores_history")
    if not user_sleep_scores_history:
        user_sleep_scores_history = [sleep_scores_history[-1].copy()]

    updated_user_sleep_scores = normalize_sleep_scores(user_sleep_scores, updated_sleep_scores.size)
    display_unchanged = np.array_equal(
        sleep_scores_history[-1], updated_sleep_scores, equal_nan=True
    )
    user_layer_unchanged = np.array_equal(
        user_sleep_scores_history[-1], updated_user_sleep_scores, equal_nan=True
    )
    if display_unchanged and user_layer_unchanged:
        return undo_button_style

    sleep_scores_history.append(updated_sleep_scores)
    user_sleep_scores_history.append(updated_user_sleep_scores)
    cache.set("sleep_scores_history", sleep_scores_history)
    cache.set("user_sleep_scores_history", user_sleep_scores_history)
    return {"visibility": "visible"}


@app.callback(
    Output("updated-sleep-scores-store", "data", allow_duplicate=True),
    Output("user-sleep-scores-store", "data", allow_duplicate=True),
    Output("undo-button", "style"),
    Input("undo-button", "n_clicks"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    sleep_scores_history = cache.get("sleep_scores_history")
    sleep_scores_history.pop()
    sleep_scores = sleep_scores_history[-1]
    user_sleep_scores_history = cache.get("user_sleep_scores_history")
    if user_sleep_scores_history:
        user_sleep_scores_history.pop()
    else:
        user_sleep_scores_history = [sleep_scores.copy()]
    cache.set("sleep_scores_history", sleep_scores_history)
    cache.set("user_sleep_scores_history", user_sleep_scores_history)
    return (
        sleep_scores.tolist(),
        user_sleep_scores_history[-1].tolist(),
        {"visibility": "hidden"},
    )


@app.callback(
    Output("annotation-message", "children", allow_duplicate=True),
    Output("interval-component", "max_intervals"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    mat_path = cache.get("filepath")
    filename = cache.get("filename")
    temp_mat_path = TEMP_PATH / f"{filename}.mat"
    mat = loadmat(mat_path, squeeze_me=True)

    # Replace None in sleep_scores
    sleep_scores_history = cache.get("sleep_scores_history")
    labels = None
    if sleep_scores_history:
        sleep_scores = sleep_scores_history[-1]
        np.place(sleep_scores, sleep_scores == None, [-1])
        sleep_scores = np.nan_to_num(sleep_scores, nan=-1)
        mat["sleep_scores"] = sleep_scores

    ne = mat.get("ne")
    if ne is not None and ne.size > 1:
        ne_standardized = standardize(ne)
        mat["ne_standardized"] = ne_standardized

    # Filter out the default keys
    mat_filtered = {}
    for key, value in mat.items():
        if not key.startswith("_"):
            mat_filtered[key] = value

    savemat(temp_mat_path, mat_filtered)

    unscored_segment = get_first_unscored_segment(mat.get("sleep_scores"))
    unscored_message = ""
    if unscored_segment is not None:
        unscored_message = (
            "Unscored segment found: "
            f"[{unscored_segment['start']}, {unscored_segment['end']}] "
            f"({unscored_segment['duration']} s). "
            "Complete scoring to export the sleep bout spreadsheet."
        )

    # Save MAT file with native dialog
    mat_save_path = save_file_dialog("mat", f"{filename}.mat")

    if not mat_save_path:
        return unscored_message, 5 if unscored_message else dash.no_update

    shutil.copy(temp_mat_path, mat_save_path)
    message = f"Saved annotations to {mat_save_path}."
    if unscored_message:
        message += f"\n{unscored_message}"

    # Export sleep bout spreadsheet only if the manual scoring is complete
    if mat.get("sleep_scores") is not None and unscored_segment is None:
        labels = mat["sleep_scores"]
        # A completely scored recording that reached disk is what "scored"
        # counts as. Re-saving or reopening it will not count it again.
        if record_scored_recording(mat):
            threading.Thread(target=sync_usage_reports, daemon=True).start()

    if labels is not None:
        labels = labels.astype(int)
        df = get_sleep_segments(labels)
        df_stats = get_pred_label_stats(df)
        temp_excel_path = TEMP_PATH / f"{filename}_table.xlsx"

        with pd.ExcelWriter(temp_excel_path) as writer:
            df.to_excel(writer, sheet_name="Sleep_bouts")
            df_stats.to_excel(writer, sheet_name="Sleep_stats")
            worksheet = writer.sheets["Sleep_stats"]
            worksheet.set_column(0, 0, 20)

        # Save Excel file with native dialog
        excel_save_path = save_file_dialog("xlsx", f"{filename}_table.xlsx")

        if excel_save_path:
            shutil.copy(temp_excel_path, excel_save_path)
            message += f"\nSaved spreadsheet to {excel_save_path}."
        else:
            message += ""

    return message, 5
