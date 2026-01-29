# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

# import os
# import json
import math
import tempfile
from collections import deque
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import dash_player
import numpy as np
import pandas as pd
import webview
from dash import Dash, Patch, clientside_callback, ctx, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask_caching import Cache
from scipy.io import loadmat, savemat

from app_src import VERSION

# from app_src.debug_tool import Debug_Counter
from app_src.components import Components
from app_src.config import POSTPROCESS
from app_src.make_figure import get_padded_sleep_scores, make_figure
from app_src.make_mp4 import make_mp4_clip
from app_src.postprocessing import get_pred_label_stats, get_sleep_segments

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

    return result[0] if result else None


def create_fig(mat, filename, default_n_shown_samples=2048):
    fig = make_figure(mat, filename, default_n_shown_samples)
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
        cache.set("sleep_scores_history", deque(maxlen=4))

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


# show_save_annotation_status
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            return [5, "Saving annotations. This may take up to 10 seconds."];
        }
        return [dash_clientside.no_update, dash_clientside.no_update];
    }
    """,
    Output("interval-component", "max_intervals"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("save-button", "n_clicks"),
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
        raise dash.exceptions.PreventUpdate

    return not is_open


@app.callback(
    Output("pred-modal-confirm", "is_open", allow_duplicate=True),
    Output("data-upload-message", "children"),
    Output("prediction-ready-store", "data"),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("save-button", "style", allow_duplicate=True),
    Output("undo-button", "style", allow_duplicate=True),
    Input("pred-confirm-button", "n_clicks"),
    State("pred-modal-confirm", "is_open"),
    prevent_initial_call=True,
)
def read_mat_pred(n_clicks, is_open):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise dash.exceptions.PreventUpdate

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
        "",
        {"visibility": "hidden"},
        {"visibility": "hidden"},
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data"),
    Output("net-annotation-count-store", "data"),
    Input("prediction-ready-store", "data"),
    State("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def generate_prediction(n_clicks, net_annotation_count):
    mat_path = cache.get("filepath")
    # filename = cache.get("filename")
    mat = loadmat(mat_path, squeeze_me=True)
    # temp_mat_path = (
    #    TEMP_PATH / f"{filename}.mat"
    # )  # savemat automatically saves as .mat file
    mat, output_path = run_inference(
        mat,
        postprocess=POSTPROCESS,
        # output_path=temp_mat_path,
    )

    sleep_scores_history = cache.get("sleep_scores_history")
    new_sleep_scores = get_padded_sleep_scores(mat)
    sleep_scores_history.append(new_sleep_scores.astype(float))
    cache.set("sleep_scores_history", sleep_scores_history)
    net_annotation_count += 1

    return "The prediction has been generated.", "pred", net_annotation_count


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("visualization-ready-store", "data", allow_duplicate=True),
    Output("upload-container", "children", allow_duplicate=True),
    Output("net-annotation-count-store", "data", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("mat-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_mat(n_clicks):
    if not n_clicks:
        raise PreventUpdate

    selected_file_path = open_file_dialog(file_type="mat")
    if selected_file_path is None:
        raise PreventUpdate  # user canceled dialog

    initialize_cache(cache, selected_file_path)
    message = "Creating visualizations... This may take up to 30 seconds."
    return message, "vis", components.mat_upload_button, 0, ""


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

    fig = create_fig(mat, filename)
    cache.set("fig_resampler", fig)
    cache.set("sleep_scores_history", sleep_scores_history)
    components.graph.figure = fig
    return components.visualization_div, metadata


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    # Output("debug-message", "children"),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
)
def update_figure(relayoutdata):
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
    return fig


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Output("net-annotation-count-store", "data", allow_duplicate=True),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),  # a keyboard press
    State("keyboard", "event"),
    State("graph", "figure"),
    State("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def add_annotation(box_select_range, keyboard_press, keyboard_event, figure, net_annotation_count):
    """update sleep scores in fig and annotation history"""
    if not (
        ctx.triggered_id == "keyboard"
        and box_select_range
        and figure["layout"]["dragmode"] == "select"
    ):
        raise PreventUpdate

    label = keyboard_event.get("key")
    if label not in ["1", "2", "3"]:
        raise PreventUpdate

    label = int(label) - 1
    start, end = box_select_range
    sleep_scores_history = cache.get("sleep_scores_history")
    current_sleep_scores = sleep_scores_history[-1]  # np array
    new_sleep_scores = current_sleep_scores.copy()
    new_sleep_scores[start:end] = label
    # If the annotation does not change anything, don't add to history
    if (new_sleep_scores == current_sleep_scores).all():
        raise PreventUpdate

    sleep_scores_history.append(new_sleep_scores.astype(float))
    cache.set("sleep_scores_history", sleep_scores_history)
    net_annotation_count += 1

    new_sleep_scores = [new_sleep_scores.tolist()]  # all numpy arrays to list for plotly 6.0 update
    patched_figure = Patch()

    for i in [-3, -2, -1]:
        # overwrite the entire z for the last 3 heatmaps
        patched_figure["data"][i]["z"] = new_sleep_scores

    # remove box or click select after an update is made
    patched_figure["layout"]["selections"] = None
    patched_figure["layout"]["shapes"] = None

    return patched_figure, "", {"visibility": "hidden"}, net_annotation_count


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Output("net-annotation-count-store", "data", allow_duplicate=True),
    Input("undo-button", "n_clicks"),
    State("graph", "figure"),
    State("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def undo_annotation(n_clicks, figure, net_annotation_count):
    sleep_scores_history = cache.get("sleep_scores_history")
    if len(sleep_scores_history) <= 1:
        raise PreventUpdate()

    net_annotation_count -= 1
    sleep_scores_history.pop()  # pop current one, then get the last one

    # undo cache
    cache.set("sleep_scores_history", sleep_scores_history)
    prev_sleep_scores = [sleep_scores_history[-1].tolist()]

    # undo figure
    patched_figure = Patch()
    for i in [-3, -2, -1]:
        # Overwrite the entire z for the last 3 heatmaps
        patched_figure["data"][i]["z"] = prev_sleep_scores

    return patched_figure, net_annotation_count


@app.callback(
    Output("save-button", "style"),
    Output("undo-button", "style"),
    Input("net-annotation-count-store", "data"),
    prevent_initial_call=True,
)
def show_hide_save_undo_button(net_annotation_count):
    sleep_scores_history = cache.get("sleep_scores_history")
    save_button_style = {"visibility": "hidden"}
    undo_button_style = {"visibility": "hidden"}
    if net_annotation_count > 0:
        save_button_style = {"visibility": "visible"}
        if len(sleep_scores_history) > 1:
            undo_button_style = {"visibility": "visible"}
    return (
        save_button_style,
        undo_button_style,
    )  # len(sleep_scores_history)


"""
@app.callback(
    Output("debug-message", "children"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)

def save_annotations(n_clicks):
    mat_path = cache.get("filepath")
    filename = cache.get("filename")
    temp_mat_path = TEMP_PATH / filename  # savemat automatically saves as .mat file
    mat = loadmat(mat_path, squeeze_me=True)
    return list(mat.keys())

"""


@app.callback(
    Output("download-annotations", "data"),
    Output("download-spreadsheet", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_path = cache.get("filepath")
    filename = cache.get("filename")
    temp_mat_path = TEMP_PATH / f"{filename}.mat"  # savemat automatically saves as .mat file
    mat = loadmat(mat_path, squeeze_me=True)

    # replace None in sleep_scores
    sleep_scores_history = cache.get("sleep_scores_history")
    labels = None
    if sleep_scores_history:
        # replace any None or nan in sleep scores to -1 before saving, otherwise results in save error
        # make a copy first because we don't want to convert any nan in the cache
        sleep_scores = sleep_scores_history[-1]
        np.place(sleep_scores, sleep_scores is None, [-1])  # convert None to -1 for scipy's savemat
        sleep_scores = np.nan_to_num(
            sleep_scores, nan=-1
        )  # convert np.nan to -1 for scipy's savemat

        mat["sleep_scores"] = sleep_scores

    # Filter out the default keys
    mat_filtered = {}
    for key, value in mat.items():
        if not key.startswith("_"):
            mat_filtered[key] = value
    savemat(temp_mat_path, mat_filtered)

    # export sleep bout spreadsheet only if the manual scoring is complete
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

        return dcc.send_file(temp_mat_path), dcc.send_file(temp_excel_path)

    return dcc.send_file(temp_mat_path), dash.no_update


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
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("reselect-video-button", "n_clicks"),
    prevent_initial_call=True,
)
def reselect_video(n_clicks):
    if n_clicks is None or n_clicks == 0:  # i.e., None or 0
        raise dash.exceptions.PreventUpdate

    message = "Please select the video above."
    return dash.no_update, components.video_upload_button, message


@app.callback(
    Output("video-path-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-upload-button", "n_clicks"),
    prevent_initial_call=True,
)
def choose_video(n_clicks):
    if not n_clicks:
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

    return player, components.reselect_video_button
