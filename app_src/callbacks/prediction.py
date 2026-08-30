# -*- coding: utf-8 -*-
"""Serverside callbacks for generating sleep-score predictions."""

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from scipy.io import loadmat

from app_src.config import POSTPROCESS, SLEEP_SCORING_MODEL
from app_src.make_figure import get_padded_sleep_scores
from app_src.run_inference_stats_model import calibrate_stats_model_config
from app_src.server import app, cache, run_inference
from app_src.sleep_score_layers import overlay_user_sleep_scores


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
    State("user-sleep-scores-store", "data"),
    prevent_initial_call=True,
)
def read_mat_pred(n_clicks, is_open, user_sleep_scores):
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
        {"user_sleep_scores": user_sleep_scores},
    )


@app.callback(
    Output("annotation-message", "children", allow_duplicate=True),
    Output("updated-sleep-scores-store", "data"),
    Input("prediction-ready-store", "data"),
    prevent_initial_call=True,
)
def generate_prediction(prediction_request):
    if not prediction_request:
        raise PreventUpdate

    user_sleep_scores = (
        prediction_request.get("user_sleep_scores")
        if isinstance(prediction_request, dict)
        else None
    )
    mat_path = cache.get("filepath")
    mat = loadmat(mat_path, squeeze_me=True)
    stats_model_config = None
    calibrated_label_count = 0
    if SLEEP_SCORING_MODEL == "stats_model":
        stats_model_config, calibrated_label_count = calibrate_stats_model_config(
            mat, user_sleep_scores
        )
    mat, output_path = run_inference(
        mat,
        postprocess=POSTPROCESS,
        stats_model_config=stats_model_config,
    )
    model_sleep_scores = get_padded_sleep_scores(mat)
    sleep_scores = overlay_user_sleep_scores(model_sleep_scores, user_sleep_scores)
    if calibrated_label_count:
        message = (
            "The adaptive statistical model was calibrated from "
            f"{calibrated_label_count} user-labelled second(s)."
        )
    else:
        message = "The prediction will be displayed shortly."
    return message, sleep_scores.tolist()
