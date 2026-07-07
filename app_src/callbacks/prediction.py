# -*- coding: utf-8 -*-
"""Serverside callbacks for generating sleep-score predictions."""

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from scipy.io import loadmat

from app_src.config import POSTPROCESS
from app_src.make_figure import get_padded_sleep_scores
from app_src.server import app, cache, run_inference


@app.callback(
    Output("pred-modal-confirm", "is_open"),
    Input("pred-button", "n_clicks"),
    State("pred-modal-confirm", "is_open"),
    prevent_initial_call=True,
)
def show_confirm_pred_modal(n_clicks, is_open):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    return not is_open


@app.callback(
    Output("pred-modal-confirm", "is_open", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("prediction-ready-store", "data"),
    Input("pred-confirm-button", "n_clicks"),
    State("pred-modal-confirm", "is_open"),
    prevent_initial_call=True,
)
def read_mat_pred(n_clicks, is_open):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    message = ""
    mat_path = cache.get("filepath")
    mat = loadmat(mat_path, squeeze_me=True)
    eeg_freq = mat["eeg_frequency"]
    if round(eeg_freq) != 512:
        message += (
            f"EEG/EMG data has a sampling frequency of {eeg_freq} Hz. Will resample to 512 Hz."
        )

    ne = mat.get("ne")
    if ne is None:
        message += " NE data not detected."

    message += (
        " Generating predictions... This may take up to 3 minutes. Check Terminal for the progress."
    )
    return (
        (not is_open),
        message,
        True,
    )


@app.callback(
    Output("annotation-message", "children", allow_duplicate=True),
    Output("updated-sleep-scores-store", "data"),
    Input("prediction-ready-store", "data"),
    prevent_initial_call=True,
)
def generate_prediction(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    mat_path = cache.get("filepath")
    mat = loadmat(mat_path, squeeze_me=True)
    mat, output_path = run_inference(
        mat,
        postprocess=POSTPROCESS,
    )
    sleep_scores = get_padded_sleep_scores(mat)
    return "The prediction will be displayed shortly.", sleep_scores.tolist()
