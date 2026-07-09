# -*- coding: utf-8 -*-
"""Serverside callbacks for loading a recording and building the figure."""

import dash
import numpy as np
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from scipy.io import loadmat

from app_src.dialogs import open_file_dialog
from app_src.make_figure import get_padded_sleep_scores
from app_src.server import app, cache, components
from app_src.session import (
    create_fig,
    find_peer_session_with_file,
    initialize_cache,
    write_metadata,
)


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
    Input("mat-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_mat(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    selected_file_path = open_file_dialog(file_type="mat")
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    if find_peer_session_with_file(selected_file_path) is not None:
        message = (
            "This file is already open in another Sleep Scoring App window. "
            "Please select a different file, or close it in the other window first."
        )
        return message, dash.no_update

    initialize_cache(cache, selected_file_path)
    message = "Creating visualizations... This may take up to 30 seconds."
    return message, "vis"


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("mat-metadata-store", "data"),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    mat_path = cache.get("filepath")
    filename = cache.get("filename")
    mat = loadmat(mat_path, squeeze_me=True)
    eeg, emg = mat.get("eeg"), mat.get("emg")

    metadata = {}
    message = "Please double check the file selected."
    validated = True
    if emg is None:
        validated = False
        message = " ".join(["EMG data is missing.", message])
    if eeg is None:
        validated = False
        message = " ".join(["EEG data is missing.", message])
    if not validated:
        return message, metadata

    metadata = write_metadata(mat)

    # salvage unsaved annotations
    sleep_scores_history = cache.get("sleep_scores_history")
    if sleep_scores_history:
        mat["sleep_scores"] = sleep_scores_history[-1]
    else:
        sleep_scores = get_padded_sleep_scores(mat)
        np.place(sleep_scores, sleep_scores == -1, [np.nan])
        sleep_scores_history.append(sleep_scores)
        cache.set("sleep_scores_history", sleep_scores_history)

    fig = create_fig(mat, filename)
    components.graph.figure = fig
    return components.visualization_div, metadata


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    sampling_level_map = {"x0.5": 1024, "x1": 2048, "x2": 4096, "x4": 8192}
    n_samples = sampling_level_map[sampling_level]
    mat_path = cache.get("filepath")
    filename = cache.get("filename")
    mat = loadmat(mat_path, squeeze_me=True)

    # copy modified (through annotation) sleep scores over
    sleep_scores_history = cache.get("sleep_scores_history")
    if sleep_scores_history:
        mat["sleep_scores"] = sleep_scores_history[-1]

    fig = create_fig(mat, filename, default_n_shown_samples=n_samples)
    # cache.set("fig_resampler", fig)
    return fig
