# -*- coding: utf-8 -*-
"""Serverside callbacks for generating sleep-score predictions."""

import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from scipy.io import loadmat

from app_src import config
from app_src.config import POSTPROCESS, SLEEP_SCORING_MODEL
from app_src.make_figure import get_padded_sleep_scores
from app_src.run_inference_stats_model import calibrate_stats_model_config
from app_src.server import app, cache, run_inference
from app_src.sleep_score_layers import overlay_user_sleep_scores, saved_user_sleep_scores
from app_src.wake_activity import WakeActivityConfig, subdivide_wake


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
    Output("prediction-message", "children", allow_duplicate=True),
    Output("prediction-ready-store", "data"),
    Output("prediction-message-interval", "n_intervals", allow_duplicate=True),
    Output("prediction-message-interval", "max_intervals", allow_duplicate=True),
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
    if round(eeg_freq) != 512 and SLEEP_SCORING_MODEL == "sdreamer":
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
        0,
        0,
    )


@app.callback(
    Output("prediction-message", "children", allow_duplicate=True),
    Output("updated-sleep-scores-store", "data"),
    Output("prediction-message-interval", "interval", allow_duplicate=True),
    Output("prediction-message-interval", "n_intervals", allow_duplicate=True),
    Output("prediction-message-interval", "max_intervals", allow_duplicate=True),
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
    if user_sleep_scores is None or len(user_sleep_scores) == 0:
        # Resolve the original saved annotations before inference replaces scores.
        # A nonempty all-null layer is intentional (e.g. after Clear), not absent.
        user_sleep_scores = saved_user_sleep_scores(mat, len(get_padded_sleep_scores(mat)))
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
    activity_result = None
    if SLEEP_SCORING_MODEL == "stats_model" and getattr(
        config, "STATS_MODEL_DETECT_WAKE_ACTIVITY", False
    ):
        try:
            activity_config = WakeActivityConfig(
                threshold=getattr(config, "WAKE_ACTIVITY_THRESHOLD", None),
                min_duration=float(getattr(config, "WAKE_ACTIVITY_MIN_DURATION", 1.0)),
                nrem_baseline_percentile=float(
                    getattr(config, "WAKE_ACTIVITY_NREM_BASELINE_PERCENTILE", 75.0)
                ),
                nrem_deviation_multiplier=float(
                    getattr(config, "WAKE_ACTIVITY_NREM_DEVIATION_MULTIPLIER", 2.0)
                ),
            )
            activity_result = subdivide_wake(
                mat.get("emg"),
                mat["eeg_frequency"],
                model_sleep_scores,
                user_sleep_scores,
                activity_config,
            )
            sleep_scores = activity_result.sleep_scores
        except (ValueError, TypeError) as error:
            return f"Wake activity detection failed: {error}", dash.no_update, 60_000, 0, 1
    if activity_result is not None:
        cache.set("wake_activity_last_run", activity_result.metadata)
    if calibrated_label_count:
        message = (
            "The adaptive statistical model was calibrated from "
            f"{calibrated_label_count} user-labelled second(s). "
            f"Tuned: Wake threshold {stats_model_config.wake_threshold:.2f}; "
            f"min Wake {stats_model_config.min_wake_duration:g} s; "
            f"min REM {stats_model_config.min_rem_duration:g} s; "
            f"global low-NE {stats_model_config.rem_threshold_percentile:g}th percentile; "
            "within-bout low-NE "
            f"{stats_model_config.rem_threshold_comparison_percentile:g}th percentile."
        )
        message_timeout_ms = 60 * 1000
    else:
        message = "The prediction will be displayed shortly."
        message_timeout_ms = 5 * 1000
    if activity_result is not None:
        message += (
            f" EMG threshold {activity_result.threshold:.6g}; "
            f"min activity {activity_config.min_duration:g} s; "
            f"{activity_result.calibrated_seconds} fine-labelled second(s)."
        )
        message_timeout_ms = 60_000
    return message, sleep_scores.tolist(), message_timeout_ms, 0, 1
