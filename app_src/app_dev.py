# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

# import os
# import json
import math
import shutil
import tempfile
from collections import deque
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_player
import numpy as np
import pandas as pd
import webview
from dash import Dash, clientside_callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from scipy.io import loadmat, savemat

from app_src import VERSION

# from app_src.debug_tool import Debug_Counter
from app_src.components_dev import Components
from app_src.config import POSTPROCESS
from app_src.make_figure_dev import get_padded_sleep_scores, make_figure
from app_src.make_mp4 import make_mp4_clip
from app_src.postprocessing import get_pred_label_stats, get_sleep_segments, normalize

try:
    from app_src.inference import run_inference

    components = Components(pred_disabled=False)
except ImportError:
    components = Components()


app = Dash(
    __name__,
    title=f"Sleep Scoring App {VERSION}",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # need this for the modal to work properly
)
app.layout = components.home_div

# debug_counter = Debug_Counter()
TEMP_PATH = Path(tempfile.gettempdir()) / "sleep_scoring_app_data"
TEMP_PATH.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Note: np.nan is converted to None when reading from cache
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 30,
        "CACHE_DEFAULT_TIMEOUT": 20
        * 24
        * 3600,  # to save cache for 20 days, otherwise it is default to 300 seconds
    },
)


# %%
def open_file_dialog(file_type):
    """
    Open a native file dialog (pywebview) with given file type filters.
    Returns a single file path as a string, or None if canceled.

    Parameters
    ----------
    file_type : tuple[str]
        Example: "mat" or "video"

    """
    if not webview.windows:
        return None

    window = webview.windows[0]

    if file_type == "mat":
        file_types = ("MAT files (*.mat)",)
    elif file_type == "video":
        file_types = ("Videos (*.avi;*.mp4)",)
    else:
        raise ValueError("Hey, it's either mat or video.")

    result = window.create_file_dialog(
        webview.FileDialog.OPEN,
        allow_multiple=False,
        file_types=file_types,
    )

    if result is None:
        return None

    if isinstance(result, (tuple, list)):
        # expected behavior on both Windows and macOS - returns tuple
        open_path = result[0] if result else None
    else:
        # In case macOS returns objc.pyobjc_unicode (string-like object)
        # See save_file_dialog
        open_path = str(result)

    return open_path


def save_file_dialog(file_type, filename):
    """
    Open a native save file dialog (pywebview) with given file type filters.
    Returns a single file path as a string, or None if canceled.

    Parameters
    ----------
    file_type : str
        Example: "mat", "xlsx", or "video"
    filename : str
        Default filename to suggest to the user

    Returns
    -------
    str or None
        The selected file path, or None if canceled
    """
    if not webview.windows:
        return None

    window = webview.windows[0]

    if file_type == "mat":
        file_types = ("MAT files (*.mat)",)
    elif file_type == "xlsx":
        file_types = ("Excel files (*.xlsx)",)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Use 'mat', 'xlsx'.")

    result = window.create_file_dialog(
        webview.FileDialog.SAVE,
        save_filename=filename,
        file_types=file_types,
    )
    # print(result)
    if result is None:
        return None

    # IMPORTANT: On macOS, SAVE dialog returns objc.pyobjc_unicode (string-like)
    # On Windows, it returns a tuple
    # Don't use result[0] - that gets the first CHARACTER, not first element!

    if isinstance(result, (tuple, list)):
        # Windows behavior - returns tuple
        save_path = result[0] if result else None
    else:
        # macOS behavior - returns objc.pyobjc_unicode (string-like object)
        save_path = str(result)  # Convert to regular Python string

    return save_path


def create_fig(mat, filename, default_n_shown_samples=2048):
    fig = make_figure(mat, filename, default_n_shown_samples)
    cache.set("fig_resampler", fig)
    return fig


def clear_temp_dir(filename):
    """clear mat and xlsx files written in temp"""
    for temp_file in TEMP_PATH.iterdir():
        if temp_file.suffix in [".mat", ".xlsx"]:
            if temp_file.stem == filename:
                continue
            temp_file.unlink()


def write_metadata(mat):
    eeg = mat.get("eeg")
    start_time = mat.get("start_time", 0)
    eeg_freq = mat.get("eeg_frequency")
    duration = math.ceil((eeg.size - 1) / eeg_freq)  # need to round duration to an int for later
    end_time = duration + start_time
    video_start_time = mat.get("video_start_time", 0)
    video_path = mat.get("video_path", "")

    if not isinstance(mat.get("video_start_time"), int):
        video_start_time = 0
    if not isinstance(video_path, str):
        video_path = ""

    metadata = dict(
        [
            ("start_time", start_time),
            ("end_time", end_time),
            ("video_start_time", video_start_time),
            ("video_path", ""),
        ]
    )
    return metadata


def initialize_cache(cache, filepath):
    cache.set("filepath", filepath)
    prev_filename = cache.get("filename")
    filename = Path(filepath).stem
    # attempt for salvaging unsaved annotations
    if prev_filename is None or prev_filename != filename:
        cache.set("sleep_scores_history", deque(maxlen=2))

    clear_temp_dir(filename)
    cache.set("filename", filename)
    recent_files_with_video = cache.get("recent_files_with_video")
    if recent_files_with_video is None:
        recent_files_with_video = []
    file_video_record = cache.get("file_video_record")
    if file_video_record is None:
        file_video_record = {}
    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)
    cache.set("fig_resampler", None)


# %% client side callbacks below

# switch_mode by pressing "m"
app.clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;

        if (key === "m" || key === "M") {
            var patched_figure = new dash_clientside.Patch;
            var predVisibility;

            if (figure.layout.dragmode === "pan") {
                // Switch to select mode
                patched_figure.assign(['layout', 'dragmode'], "select");
                predVisibility = {"visibility": "visible"};
            } else if (figure.layout.dragmode === "select") {
                // Switch to pan mode and clear selections
                patched_figure.assign(['layout', 'selections'], null);
                patched_figure.assign(['layout', 'shapes'], null);
                patched_figure.assign(['layout', 'dragmode'], "pan");
                predVisibility = {"visibility": "hidden"};
            }

            return [patched_figure.build(), "", {"visibility": "hidden"}, predVisibility];
        }

        return [dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure"),
    Output("annotation-message", "children"),
    Output("video-button", "style"),
    Output("pred-button", "style"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
)

# pan_figure
clientside_callback(
    """
    function(keyboard_nevents, keyboard_event, relayoutdata, figure) {
        if (!keyboard_event || !figure) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }

        var key = keyboard_event.key;
        var xaxisRange = figure.layout.xaxis4.range;
        var x0 = xaxisRange[0];
        var x1 = xaxisRange[1];
        var newRange;

        if (key === "ArrowRight") {
            newRange = [x0 + (x1 - x0) * 0.3, x1 + (x1 - x0) * 0.3];
        } else if (key === "ArrowLeft") {
            newRange = [x0 - (x1 - x0) * 0.3, x1 - (x1 - x0) * 0.3];
        }

        if (newRange) {
            // Use Patch for efficient partial update
            var patched_figure = new dash_clientside.Patch;
            patched_figure.assign(['layout', 'xaxis4', 'range'], newRange);

            // Create NEW object instead of mutating
            var newRelayoutData = {
                ...relayoutdata,  // Spread existing properties
                'xaxis4.range[0]': newRange[0],
                'xaxis4.range[1]': newRange[1]
            };

            return [patched_figure.build(), newRelayoutData];
        }

        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "relayoutData"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "relayoutData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# clear_display
clientside_callback(
    """
    function(n_intervals) {
        return n_intervals === 5 ? "" : dash_clientside.no_update;
    }
    """,
    Output("annotation-message", "children", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)


# read_box_select
app.clientside_callback(
    """
    function(box_select, figure, clickData, metadata) {
        // Return no_update for all outputs if conditions not met
        const no_update = dash_clientside.no_update;

        if (!figure || !metadata) {
            return [no_update, no_update, no_update, no_update];
        }

        const video_button_style = {"visibility": "hidden"};
        const selections = figure.layout.selections;

        // When selections is None/undefined, prevent update
        if (!selections || selections.length === 0) {
            return [no_update, no_update, no_update, no_update];
        }

        // Clone figure to avoid mutating state
        var patched_figure = new dash_clientside.Patch;

        // Allow only at most one select box in all subplots
        if (selections.length > 1) {
            patched_figure.assign(['layout', 'selections'], [selections[selections.length - 1]]);
        }

        // Remove existing click select box if any
        patched_figure.assign(['layout', 'shapes'], null);

        const selection = selections[selections.length - 1];

        // Take the min as start and max as end
        let start = Math.min(selection.x0, selection.x1);
        let end = Math.max(selection.x0, selection.x1);

        const eeg_start_time = metadata.start_time;
        const eeg_end_time = metadata.end_time;

        // Check if out of range
        if (end < eeg_start_time || start > eeg_end_time) {
            return [
                [],
                patched_figure.build(),
                `Out of range. Please select from ${eeg_start_time} to ${eeg_end_time}.`,
                video_button_style
            ];
        }

        // Round start and end
        let start_round = Math.round(start);
        let end_round = Math.round(end);

        start_round = Math.max(start_round, eeg_start_time);
        end_round = Math.min(end_round, eeg_end_time);

        // Handle case where start_round equals end_round
        if (start_round === end_round) {
            if (start_round - start > end - end_round) {
                // Spanning over two consecutive seconds
                end_round = Math.ceil(start);
                start_round = Math.floor(start);
            } else {
                end_round = Math.ceil(end);
                start_round = Math.floor(end);
            }
        }

        // Adjust relative to eeg_start_time
        const final_start = start_round - eeg_start_time;
        const final_end = end_round - eeg_start_time;

        // Show video button if valid range
        let final_video_button_style = {"visibility": "hidden"};
        if (final_end - final_start >= 1 && final_end - final_start <= 300) {
            final_video_button_style = {"visibility": "visible"};
        }

        return [
            [final_start, final_end],
            patched_figure.build(),
            `You selected [${final_start}, ${final_end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.`,
            final_video_button_style
        ];
    }
    """,
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    State("graph", "clickData"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)

# read_click_select
app.clientside_callback(
    """
    function(clickData, figure, metadata) {
        const no_update = dash_clientside.no_update;

        if (!figure || !metadata) {
            return [no_update, no_update, no_update, no_update];
        }

        // Clone figure to avoid mutating state
        var patched_figure = new dash_clientside.Patch;

        // Remove existing select box if any
        patched_figure.assign(['layout', 'shapes'], null);

        const video_button_style = {"visibility": "hidden"};
        const dragmode = figure.layout.dragmode;

        // If no click data or in pan mode, return defaults
        if (!clickData || dragmode === "pan") {
            return [[], patched_figure.build(), "", video_button_style];
        }

        // Remove the box selection if present
        patched_figure.assign(['layout', 'selections'], null);

        // Grab clicked x value
        const x_click = clickData.points[0].x;

        // Determine current x-axis visible range
        const x_min = figure.layout.xaxis4.range[0];
        const x_max = figure.layout.xaxis4.range[1];
        const total_range = x_max - x_min;

        // Decide neighborhood size: 0.5% of current view range
        const fraction = 0.005;
        const delta = total_range * fraction;

        const eeg_start_time = metadata.start_time;
        const eeg_end_time = metadata.end_time;

        const x0 = Math.floor(x_click - delta / 2);
        const x1 = Math.ceil(x_click + delta / 2);

        // Get curve information
        const curve_index = clickData.points[0].curveNumber;
        const trace = figure.data[curve_index];
        const xref = trace.xaxis || "x4";  // x4 is the shared x-axis
        let yref = trace.yaxis || "y5";    // spectrogram has dual y-axis

        // Use the left y-axis to avoid interfering with theta/delta curve
        if (yref === "y2") {
            yref = "y1";
        }

        // Create select box
        const select_box = {
            "type": "rect",
            "xref": xref,
            "yref": yref,
            "x0": x0,
            "x1": x1,
            "y0": -30,
            "y1": 30,
            "line": {"width": 1, "dash": "dot"}
        };

        patched_figure.assign(['layout', 'shapes'], [select_box]);

        // Calculate final start and end
        const start = Math.max(x0, eeg_start_time);
        const end = Math.min(x1, eeg_end_time);

        // Show video button if valid range
        let final_video_button_style = {"visibility": "hidden"};
        if (end - start >= 1 && end - start <= 300) {
            final_video_button_style = {"visibility": "visible"};
        }

        return [
            [start, end],
            patched_figure.build(),
            `You selected [${start}, ${end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.`,
            final_video_button_style
        ];
    }
    """,
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "clickData"),
    State("graph", "figure"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)

# update_sleep_scores
app.clientside_callback(
    """
    function(sleep_scores, figure) {
        const no_update = dash_clientside.no_update;

        if (!sleep_scores || !Array.isArray(sleep_scores) || !figure) {
            return [no_update, no_update];
        }

        // Use Patch for efficient update
        var patched_figure = new dash_clientside.Patch;

        // Wrap in array for heatmap z-data format
        const sleep_scores_wrapped = [sleep_scores];

        // Calculate actual indices (last 3 traces)
        const num_traces = figure.data.length;
        const indices = [num_traces - 3, num_traces - 2, num_traces - 1];

        // Update all 3 heatmaps
        for (const idx of indices) {
            patched_figure.assign(['data', idx, 'z'], sleep_scores_wrapped);
        }

        // Clear selections
        patched_figure.assign(['layout', 'selections'], null);
        patched_figure.assign(['layout', 'shapes'], null);

        return [patched_figure.build(), ""];
    }
    """,
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("updated-sleep-scores-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# make_annotation
app.clientside_callback(
    """
    function(keyboard_press, keyboard_event, box_select_range, figure) {
        const no_update = dash_clientside.no_update;

        // Only proceed if we have all required data
        if (!keyboard_event || !box_select_range || box_select_range.length === 0 || !figure) {
            return [no_update, no_update, no_update];
        }

        // Check if in select mode
        if (figure.layout.dragmode !== "select") {
            return [no_update, no_update, no_update];
        }

        const label = keyboard_event.key;
        if (!["1", "2", "3"].includes(label)) {
            return [no_update, no_update, no_update];
        }

        const label_int = parseInt(label) - 1;
        const [start, end] = box_select_range;

        // Get current sleep scores from last trace
        const last_trace = figure.data[figure.data.length - 1];
        const current_sleep_scores = last_trace.z[0];

        // Create a copy using spread operator
        const sleep_scores = [...current_sleep_scores];

        // Update the range
        for (let i = start; i < end; i++) {
            sleep_scores[i] = label_int;
        }

        return [
            {"visibility": "hidden"},
            sleep_scores,
            []  // Clear box selection after annotation
        ];
    }
    """,
    Output("video-button", "style", allow_duplicate=True),
    Output("updated-sleep-scores-store", "data", allow_duplicate=True),
    Output("box-select-store", "data", allow_duplicate=True),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("box-select-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)

# %% server side callbacks below

"""
@app.callback(
    Output("debug-message", "children"),
    Input("box-select-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def debug_box_select(box_select, figure):
    #time_end = figure["data"][-1]["z"][0][-1]
    return json.dumps(figure["data"][-1]["z"], indent=2)
"""


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
    # Output("debug-message", "children"),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_fig_resampler(relayoutdata):
    if relayoutdata is None:
        return dash.no_update

    if "xaxis4.range[0]" not in relayoutdata and "xaxis4.range" not in relayoutdata:
        return dash.no_update

    fig = cache.get("fig_resampler")
    if fig is None:
        return dash.no_update

    # debug_counter.increment()
    return fig.construct_update_data_patch(relayoutdata)


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    if sampling_level is None:
        return dash.no_update
    sampling_level_map = {"x1": 2048, "x2": 4096, "x4": 8192}
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


@app.callback(
    Output("updated-sleep-scores-store", "data", allow_duplicate=True),
    Output("undo-button", "style"),
    Input("undo-button", "n_clicks"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    sleep_scores_history = cache.get("sleep_scores_history")
    sleep_scores = sleep_scores_history[0]
    sleep_scores_history.pop()
    cache.set("sleep_scores_history", sleep_scores_history)
    return sleep_scores.tolist(), {"visibility": "hidden"}


@app.callback(
    Output("undo-button", "style", allow_duplicate=True),
    Input("updated-sleep-scores-store", "data"),
    prevent_initial_call=True,
)
def update_sleep_scores_history(updated_sleep_scores):
    """
    always starts with at least one list when a mat file is read

    """
    undo_button_style = {"visibility": "hidden"}
    if not updated_sleep_scores:
        return undo_button_style

    sleep_scores_history = cache.get("sleep_scores_history")
    updated_sleep_scores = np.array(updated_sleep_scores, dtype=float)
    if np.array_equal(sleep_scores_history[-1], updated_sleep_scores, equal_nan=True):  # no change
        return undo_button_style

    sleep_scores_history.append(updated_sleep_scores)
    cache.set("sleep_scores_history", sleep_scores_history)
    return {"visibility": "visible"}


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
        np.place(sleep_scores, sleep_scores is None, [-1])
        sleep_scores = np.nan_to_num(sleep_scores, nan=-1)
        mat["sleep_scores"] = sleep_scores

    ne = mat.get("ne")
    if ne is not None and ne.size > 1:
        ne_normalized = normalize(ne)
        mat["ne_normalized"] = ne_normalized
        
    # Filter out the default keys
    mat_filtered = {}
    for key, value in mat.items():
        if not key.startswith("_"):
            mat_filtered[key] = value

    savemat(temp_mat_path, mat_filtered)

    # Save MAT file with native dialog
    mat_save_path = save_file_dialog("mat", f"{filename}.mat")

    if not mat_save_path:
        return "", dash.no_update

    shutil.copy(temp_mat_path, mat_save_path)
    message = f"Saved annotations to {mat_save_path}."

    # Export sleep bout spreadsheet only if the manual scoring is complete
    if mat.get("sleep_scores") is not None and -1 not in mat["sleep_scores"]:
        labels = mat["sleep_scores"]

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


@app.callback(
    Output("video-modal", "is_open"),
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-button", "n_clicks"),
    State("video-modal", "is_open"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)
def prepare_video(n_clicks, is_open, metadata):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    file_unseen = True
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    if filename in recent_files_with_video:
        recent_files_with_video.remove(filename)
        video_info = file_video_record.get(filename)
        if video_info is not None and Path(video_info["video_path"]).is_file():
            file_unseen = False

    recent_files_with_video.append(filename)
    cache.set("recent_files_with_video", recent_files_with_video)
    if not file_unseen:
        video_path = video_info["video_path"]
        message = "Preparing clip..."
        return (not is_open), video_path, "", message

    # if original avi has not been uploaded, ask for it
    # video_path = cache.get("video_path")
    video_path = metadata.get("video_path")
    message = "Please select the video above."
    if video_path:
        message += f" You may find it at {video_path}."
    return (not is_open), dash.no_update, components.video_upload_button, message


@app.callback(
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-title", "children"),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("reselect-video-button", "n_clicks"),
    prevent_initial_call=True,
)
def reselect_video(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    message = "Please select video above."
    return dash.no_update, "", components.video_upload_button, message


@app.callback(
    Output("video-path-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_video(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise PreventUpdate

    selected_file_path = open_file_dialog(file_type="video")
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    avi_path = Path(selected_file_path)  # need to turn WindowsPath to str for the output
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    file_video_record[filename] = {
        "video_path": str(avi_path),
        "video_name": avi_path.name,
    }
    if len(recent_files_with_video) > 3:
        filename_to_remove = recent_files_with_video.pop(0)
        if filename_to_remove in file_video_record:
            # avi_file_to_remove = Path(
            #    file_video_record[filename_to_remove]["video_path"]
            # )
            file_video_record.pop(filename_to_remove)
            # avi_file_to_remove.unlink(missing_ok=False)

    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)

    return str(avi_path), "Preparing clip..."


@app.callback(
    Output("clip-name-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-path-store", "data"),
    State("box-select-store", "data"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)
def make_clip(video_path, box_select_range, metadata):
    if not box_select_range:
        return dash.no_update, ""

    start, end = box_select_range
    video_start_time = metadata.get("video_start_time")
    start = start + video_start_time
    end = end + video_start_time
    video_name = Path(video_path).stem
    clip_name = video_name + f"_time_range_{start}-{end}" + ".mp4"
    save_path = VIDEO_DIR / clip_name
    if save_path.is_file():
        return clip_name, ""

    for file in VIDEO_DIR.iterdir():
        if file.is_file() and file.suffix == ".mp4":
            file.unlink()

    try:
        make_mp4_clip(
            video_path,
            start_time=start,
            end_time=end,
            save_path=save_path,
        )
    except ValueError as error_message:
        return dash.no_update, repr(error_message)

    return clip_name, ""


@app.callback(
    Output("video-title", "children", allow_duplicate=True),
    Output("video-container", "children"),
    Output("video-message", "children"),
    Input("clip-name-store", "data"),
    prevent_initial_call=True,
)
def show_clip(clip_name):
    if not (VIDEO_DIR / clip_name).is_file():
        return "", "Video not ready yet. Please check again in a second."

    clip_path = Path("/assets/videos") / clip_name
    player = dash_player.DashPlayer(
        id="player",
        url=str(clip_path),
        controls=True,
        width="100%",
        height="100%",
    )

    return clip_name, player, components.reselect_video_button
