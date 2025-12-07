# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import math

import tempfile
import webbrowser
from pathlib import Path
from collections import deque

import dash
import dash_player
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash import Dash, dcc, ctx, clientside_callback, Patch

import numpy as np
import pandas as pd
from flask_caching import Cache
from scipy.io import loadmat, savemat

from app_src import VERSION, config
from app_src.make_mp4 import make_mp4_clip
from app_src.components_dev import Components
from app_src.make_figure_dev import get_padded_sleep_scores, make_figure

from app_src.postprocessing import get_sleep_segments, get_pred_label_stats


app = Dash(
    __name__,
    title=f"Sleep Scoring App {VERSION}",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP
    ],  # need this for the modal to work properly
)

TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

VIDEO_DIR = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

try:
    from app_src.inference import run_inference

    components = Components(pred_disabled=False)
except ImportError:
    components = Components()

app.layout = components.home_div
du = components.configure_du(app, TEMP_PATH)

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
def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


def create_fig(mat, mat_name, default_n_shown_samples=2048):
    fig = make_figure(mat, mat_name, default_n_shown_samples)
    return fig


def initialize_cache(cache, filename):
    prev_filename = cache.get("filename")

    # attempt for salvaging unsaved annotations
    if prev_filename is None or prev_filename != filename:
        cache.set("sleep_scores_history", deque(maxlen=4))

    cache.set("filename", filename)
    recent_files_with_video = cache.get("recent_files_with_video")
    if recent_files_with_video is None:
        recent_files_with_video = []
    file_video_record = cache.get("file_video_record")
    if file_video_record is None:
        file_video_record = {}
    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)
    cache.set("start_time", 0)
    cache.set("end_time", 0)
    cache.set("video_start_time", 0)
    cache.set("video_name", "")
    cache.set("video_path", "")
    cache.set("fig_resampler", None)


def update_cache(mat):
    eeg = mat.get("eeg")
    start_time = mat.get("start_time", 0)
    eeg_freq = mat.get("eeg_frequency")
    duration = math.ceil(
        (eeg.size - 1) / eeg_freq
    )  # need to round duration to an int for later
    end_time = duration + start_time
    video_start_time = mat.get("video_start_time", 0)
    video_path = mat.get("video_path", "")
    video_name = mat.get("video_name", "")
    cache.set("start_time", start_time)
    cache.set("end_time", end_time)
    if video_start_time is not None:
        cache.set("video_start_time", video_start_time)
    if video_path:
        cache.set("video_path", video_path)
    if video_name:
        cache.set("video_name", video_name)


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
            let updatedFigure = JSON.parse(JSON.stringify(figure));
            if (figure.layout.dragmode === "pan") {
                updatedFigure.layout.dragmode = "select"
                predVisibility = {"visibility": "visible"}
            } else if (figure.layout.dragmode === "select") {
                updatedFigure.layout.selections = null;
                updatedFigure.layout.shapes = null;
                updatedFigure.layout.dragmode = "pan"
                predVisibility = {"visibility": "hidden"}
            }
            return [updatedFigure, "", {"visibility": "hidden"}, predVisibility];
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

# pan_figures
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
            relayoutdata['xaxis4.range[0]'] = newRange[0];
            relayoutdata['xaxis4.range[1]'] = newRange[1];
            figure.layout.xaxis4.range = newRange;
            return [figure, relayoutdata];
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

# %% server side callbacks below


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
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)
    eeg_freq = mat["eeg_frequency"]
    if round(eeg_freq) != 512:
        message += (
            f"EEG/EMG data has a sampling frequency of {eeg_freq} Hz. "
            "Will resample to 512 Hz."
        )

    ne = mat.get("ne")
    if ne is None:
        message += " NE data not detected."

    message += " Generating predictions... This may take up to 3 minutes. Check Terminal for the progress."
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
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)
    filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, filename)
    mat, output_path = run_inference(
        mat,
        postprocess=config["postprocess"],
        output_path=temp_mat_path,
    )

    sleep_scores_history = cache.get("sleep_scores_history")
    # new_sleep_scores = mat.get("sleep_scores")
    new_sleep_scores = get_padded_sleep_scores(mat)
    sleep_scores_history.append(new_sleep_scores.astype(float))
    cache.set("sleep_scores_history", sleep_scores_history)
    net_annotation_count += 1

    return "The prediction has been generated.", "pred", net_annotation_count


@du.callback(
    output=[
        Output("data-upload-message", "children", allow_duplicate=True),
        Output("visualization-ready-store", "data", allow_duplicate=True),
        Output("upload-container", "children", allow_duplicate=True),
        Output("net-annotation-count-store", "data", allow_duplicate=True),
        Output("annotation-message", "children", allow_duplicate=True),
    ],
    id="vis-data-upload",
)
def read_mat_vis(status):
    # clean TEMP_PATH regularly by deleting temp files written there
    mat_file = Path(status.latest_file)
    filename = mat_file.stem

    temp_dir = Path(TEMP_PATH)
    for temp_file in temp_dir.iterdir():
        if temp_file.suffix in [".mat", ".xlsx"]:
            if temp_file.stem == filename:
                continue
            temp_file.unlink()

    initialize_cache(cache, mat_file.name)
    message = (
        "File uploaded. Creating visualizations... This may take up to 30 seconds."
    )
    return message, "vis", components.vis_upload_box, 0, ""


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    # Output("debug-message", "children"),
    Input("visualization-ready-store", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)
    eeg, emg = mat.get("eeg"), mat.get("emg")

    message = "Please double check the file selected."
    validated = True
    if emg is None:
        validated = False
        message = " ".join(["EMG data is missing.", message])
    if eeg is None:
        validated = False
        message = " ".join(["EEG data is missing.", message])
    if not validated:
        return message

    update_cache(mat)

    # salvage unsaved annotations
    sleep_scores_history = cache.get("sleep_scores_history")
    if sleep_scores_history:
        mat["sleep_scores"] = sleep_scores_history[-1]
    else:
        sleep_scores = get_padded_sleep_scores(mat)
        np.place(sleep_scores, sleep_scores == -1, [np.nan])
        sleep_scores_history.append(sleep_scores)

    fig = create_fig(mat, mat_name)
    cache.set("fig_resampler", fig)
    cache.set("sleep_scores_history", sleep_scores_history)
    components.graph.figure = fig
    return components.visualization_div


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
    mat_name = cache.get("filename")
    mat = loadmat(os.path.join(TEMP_PATH, mat_name), squeeze_me=True)

    # copy modified (through annotation) sleep scores over
    sleep_scores_history = cache.get("sleep_scores_history")
    if sleep_scores_history:
        mat["sleep_scores"] = sleep_scores_history[-1]

    fig = create_fig(mat, mat_name, default_n_shown_samples=n_samples)
    return fig


@app.callback(
    Output("video-modal", "is_open"),
    Output("video-path-store", "data", allow_duplicate=True),
    Output("video-container", "children", allow_duplicate=True),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-button", "n_clicks"),
    State("video-modal", "is_open"),
    prevent_initial_call=True,
)
def prepare_video(n_clicks, is_open):
    file_unseen = True
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    if filename in recent_files_with_video:
        recent_files_with_video.remove(filename)
        video_info = file_video_record.get(filename)
        if video_info is not None and os.path.isfile(video_info["video_path"]):
            file_unseen = False

    recent_files_with_video.append(filename)
    cache.set("recent_files_with_video", recent_files_with_video)
    if not file_unseen:
        video_path = video_info["video_path"]
        message = "Preparing clip..."
        return (not is_open), video_path, "", message

    # if original avi has not been uploaded, ask for it
    video_path = cache.get("video_path")
    message = "Please upload the original video above."
    if video_path:
        message += f" You may find it at {video_path}."
    return (not is_open), dash.no_update, components.video_upload_box, message


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

    message = "Please upload the original video above."
    return dash.no_update, components.video_upload_box, message


@du.callback(
    output=[
        Output("video-path-store", "data"),
        Output("video-message", "children", allow_duplicate=True),
    ],
    id="video-upload",
)
def upload_video(status):
    avi_path = status.latest_file  # a WindowsPath
    avi_path = str(avi_path)  # need to turn WindowsPath to str for the output
    filename = cache.get("filename")
    recent_files_with_video = cache.get("recent_files_with_video")
    file_video_record = cache.get("file_video_record")
    file_video_record[filename] = {
        "video_path": avi_path,
        "video_name": os.path.basename(avi_path),
    }
    if len(recent_files_with_video) > 3:
        filename_to_remove = recent_files_with_video.pop(0)
        if filename_to_remove in file_video_record:
            avi_file_to_remove = file_video_record[filename_to_remove]["video_path"]
            file_video_record.pop(filename_to_remove)
            if os.path.isfile(avi_file_to_remove):
                os.remove(avi_file_to_remove)

    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)

    return avi_path, "Preparing clip..."


@app.callback(
    Output("clip-name-store", "data"),
    Output("video-message", "children", allow_duplicate=True),
    Input("video-path-store", "data"),
    State("box-select-store", "data"),
    prevent_initial_call=True,
)
def make_clip(video_path, box_select_range):
    if not box_select_range:
        return dash.no_update, ""

    start, end = box_select_range
    video_start_time = cache.get("video_start_time")
    # start_time = cache.get("start_time")
    start = start + video_start_time
    end = end + video_start_time
    video_name = os.path.basename(video_path).split(".")[0]
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
    clip_path = os.path.join("/assets/videos/", clip_name)
    player = dash_player.DashPlayer(
        id="player",
        url=clip_path,
        controls=True,
        width="100%",
        height="100%",
    )

    return player, components.reselect_video_button


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("graph", "relayoutData"),
    prevent_initial_call=True,
    memoize=True,
)
def update_fig(relayoutdata):
    fig = cache.get("fig_resampler")
    if fig is None:
        return dash.no_update

    # manually supply xaxis4.range[0] and xaxis4.range[1] after clicking
    # reset axes button because it only gives xaxis4.range. It seems
    # updating fig_resampler requires xaxis4.range[0] and xaxis4.range[1]
    if (
        relayoutdata.get("xaxis4.range") is not None
        and relayoutdata.get("xaxis4.range[0]") is None
    ):
        relayoutdata["xaxis4.range[0]"], relayoutdata["xaxis4.range[1]"] = relayoutdata[
            "xaxis4.range"
        ]
    return fig.construct_update_data_patch(relayoutdata)


@app.callback(
    # Output("debug-message", "children"),
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    State("graph", "clickData"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure, clickData):
    video_button_style = {"visibility": "hidden"}
    selections = figure["layout"].get("selections")

    # when selections is None, it means there's not box select in the graph
    if selections is None:
        raise PreventUpdate()

    # allow only at most one select box in all subplots
    if len(selections) > 1:
        selections.pop(0)

    patched_figure = Patch()
    patched_figure["layout"][
        "selections"
    ] = selections  # patial property update: https://dash.plotly.com/partial-properties#update
    patched_figure["layout"]["shapes"] = None  # remove click select box

    # take the min as start and max as end so that how the box is drawn doesn't matter
    start, end = min(selections[0]["x0"], selections[0]["x1"]), max(
        selections[0]["x0"], selections[0]["x1"]
    )
    eeg_start_time = cache.get("start_time")
    eeg_end_time = cache.get("end_time")

    if end < eeg_start_time or start > eeg_end_time:
        return (
            [],
            patched_figure,
            f"Out of range. Please select from {eeg_start_time} to {eeg_end_time}.",
            video_button_style,
        )

    start_round, end_round = round(start), round(end)
    start_round = max(start_round, eeg_start_time)
    end_round = min(end_round, eeg_end_time)
    if start_round == end_round:
        if (
            start_round - start > end - end_round
        ):  # spanning over two consecutive seconds
            end_round = math.ceil(start)
            start_round = math.floor(start)
        else:
            end_round = math.ceil(end)
            start_round = math.floor(end)

    start, end = start_round - eeg_start_time, end_round - eeg_start_time
    if 1 <= end - start <= 300:
        video_button_style = {"visibility": "visible"}

    return (
        [start, end],
        patched_figure,
        f"You selected [{start}, {end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.",
        video_button_style,
    )


"""
@app.callback(
    Output("debug-message", "children"),
    Input("box-select-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def debug_box_select(box_select, figure):
    #time_end = figure["data"][-1]["z"][0][-1]
    return json.dumps(box_select, indent=2)
"""


@app.callback(
    # Output("debug-message", "children", allow_duplicate=True),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "clickData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_click_select(clickData, figure):  # triggered only  if clicked within x-range
    patched_figure = Patch()
    patched_figure["layout"]["shapes"] = None
    video_button_style = {"visibility": "hidden"}
    dragmode = figure["layout"]["dragmode"]
    if clickData is None or dragmode == "pan":
        return [], patched_figure, "", video_button_style

    # remove the select box if present
    patched_figure["layout"]["selections"] = None

    # Grab clicked x value
    x_click = clickData["points"][0]["x"]

    # Determine current x-axis visible range
    x_min, x_max = figure["layout"]["xaxis4"]["range"]
    total_range = x_max - x_min

    # Decide neighborhood size: e.g., 1% of current view range
    fraction = 0.005  # 0.5% (adjustable)
    delta = total_range * fraction
    eeg_start_time = cache.get("start_time")
    eeg_end_time = cache.get("end_time")
    x0, x1 = math.floor(x_click - delta / 2), math.ceil(x_click + delta / 2)
    curve_index = clickData["points"][0]["curveNumber"]
    trace = figure["data"][curve_index]
    xref = trace.get("xaxis", "x4")  # x4 is the shared x-axis
    yref = trace.get("yaxis", "y5")  # spectrogram has dual y-axis

    if yref == "y2":  # use the left y-axis to avoid interfering with theta/delta curve
        yref = "y1"

    select_box = {
        "type": "rect",
        "xref": xref,
        "yref": yref,
        "x0": x0,
        "x1": x1,
        "y0": -20,
        "y1": 20,
        "line": {"width": 1, "dash": "dot"},
    }

    patched_figure["layout"]["shapes"] = [select_box]
    start = max(x0, eeg_start_time)
    end = min(x1, eeg_end_time)

    if 1 <= end - start <= 300:
        video_button_style = {"visibility": "visible"}
    return (
        [start, end],
        patched_figure,
        f"You selected [{start}, {end}]. Press 1 for Wake, 2 for NREM, or 3 for REM.",
        video_button_style,
    )


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    # Output("annotation-store", "data"),
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
def update_sleep_scores(
    box_select_range, keyboard_press, keyboard_event, figure, net_annotation_count
):
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
    new_sleep_scores[start:end] = np.array([label] * (end - start))
    # If the annotation does not change anything, don't add to history
    if (new_sleep_scores == current_sleep_scores).all():
        raise PreventUpdate

    sleep_scores_history.append(new_sleep_scores.astype(float))
    cache.set("sleep_scores_history", sleep_scores_history)
    net_annotation_count += 1

    patched_figure = Patch()
    patched_figure["data"][-3]["z"][0] = new_sleep_scores
    patched_figure["data"][-2]["z"][0] = new_sleep_scores
    patched_figure["data"][-1]["z"][0] = new_sleep_scores

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
    prev_sleep_scores = sleep_scores_history[-1]

    # undo figure
    patched_figure = Patch()
    patched_figure["data"][-3]["z"][0] = prev_sleep_scores
    patched_figure["data"][-2]["z"][0] = prev_sleep_scores
    patched_figure["data"][-1]["z"][0] = prev_sleep_scores
    return patched_figure, net_annotation_count


@app.callback(
    Output("save-button", "style"),
    Output("undo-button", "style"),
    # Output("debug-message", "children"),
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


@app.callback(
    Output("download-annotations", "data"),
    Output("download-spreadsheet", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = loadmat(temp_mat_path, squeeze_me=True)

    # replace None in sleep_scores
    sleep_scores_history = cache.get("sleep_scores_history")
    labels = None
    if sleep_scores_history:
        # replace any None or nan in sleep scores to -1 before saving, otherwise results in save error
        # make a copy first because we don't want to convert any nan in the cache
        sleep_scores = sleep_scores_history[-1]
        np.place(
            sleep_scores, sleep_scores == None, [-1]
        )  # convert None to -1 for scipy's savemat
        sleep_scores = np.nan_to_num(
            sleep_scores, nan=-1
        )  # convert np.nan to -1 for scipy's savemat

        mat["sleep_scores"] = sleep_scores
    savemat(temp_mat_path, mat)

    # export sleep bout spreadsheet only if the manual scoring is complete
    if mat.get("sleep_scores") is not None and -1 not in mat["sleep_scores"]:
        labels = mat["sleep_scores"]

    if labels is not None:
        labels = labels.astype(int)
        df = get_sleep_segments(labels)
        df_stats = get_pred_label_stats(df)
        temp_excel_path = os.path.splitext(temp_mat_path)[0] + "_table.xlsx"
        with pd.ExcelWriter(temp_excel_path) as writer:
            df.to_excel(writer, sheet_name="Sleep_bouts")
            df_stats.to_excel(writer, sheet_name="Sleep_stats")
            worksheet = writer.sheets["Sleep_stats"]
            worksheet.set_column(0, 0, 20)

        return dcc.send_file(temp_mat_path), dcc.send_file(temp_excel_path)

    return dcc.send_file(temp_mat_path), dash.no_update


if __name__ == "__main__":
    from threading import Timer
    from functools import partial

    PORT = 8050
    Timer(1, partial(open_browser, PORT)).start()
    app.run_server(debug=False, port=PORT, dev_tools_hot_reload=False)
