# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""

import os
import json
import base64
import tempfile
import webbrowser
from io import BytesIO
from threading import Timer
from collections import deque

import dash
from dash import Dash, dcc, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from dash_extensions import EventListener

from flask_caching import Cache
from trace_updater import TraceUpdater
from plotly_resampler import FigureResampler

from scipy.io import loadmat, savemat

from inference import run_inference
from make_figure import make_figure
from components import mat_upload_box, graph, visualization_div


app = Dash(__name__)
port = 8050

TEMP_PATH = os.path.join(tempfile.gettempdir(), "sleep_scoring_app_data")
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 10,
        #'CACHE_DEFAULT_TIMEOUT': 86400, # to save cache for 1 day
    },
)


def create_fig(default_n_shown_samples=2000):
    fig = FigureResampler(default_n_shown_samples=default_n_shown_samples)
    fig.register_update_graph_callback(
        app=app, graph_id="graph", trace_updater_id="trace-updater"
    )
    return fig


def initiate_cache(cache):
    cache.set("filename", "")
    cache.set("mat", None)
    cache.set("annotation_history", deque(maxlen=3))
    cache.set("updated_pred", None)
    cache.set("updated_conf", None)


def run_app():
    app.layout = html.Div(
        [
            dcc.RadioItems(
                id="task-selection",
                options=[
                    {"label": "Generate prediction", "value": "gen"},
                    {"label": "Visualize existing prediction", "value": "vis"},
                ],
            ),
            html.Div(id="upload-container"),
            html.Div(id="data-upload-message"),
            dcc.Store(id="extension-validation"),
            dcc.Store(id="generation-ready"),
            dcc.Store(id="visualization-ready"),
            dcc.Download(id="prediction-download"),
        ]
    )


@app.callback(Output("upload-container", "children"), Input("task-selection", "value"))
def show_upload(task):
    if task is None:
        raise PreventUpdate
    else:
        return mat_upload_box


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("extension-validation", "data"),
    Input(mat_upload_box, "contents"),
    State(mat_upload_box, "filename"),
    prevent_initial_call=True,
)
def validate_extension(contents, filename):
    if contents is None:
        return html.Div(["Please select a .mat file."]), dash.no_update

    if not filename.endswith(".mat"):
        return html.Div(["Please select a .mat file only."]), dash.no_update

    return (
        html.Div(["Loading and validating file... This may take up to 10 seconds."]),
        True,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("generation-ready", "data"),
    Output("visualization-ready", "data"),
    Input("extension-validation", "data"),
    State(mat_upload_box, "contents"),
    State(mat_upload_box, "filename"),
    State("task-selection", "value"),
    prevent_initial_call=True,
)
def read_mat(extension_validated, contents, filename, task):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    mat = loadmat(BytesIO(decoded))
    if mat.get("trial_eeg") is None:
        return (
            html.Div(["EEG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    if mat.get("trial_emg") is None:
        return (
            html.Div(["EMG data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    if mat.get("trial_ne") is None:
        return (
            html.Div(["NE data is missing. Please double check the file selected."]),
            dash.no_update,
            dash.no_update,
        )

    initiate_cache(cache)
    cache.set("filename", filename)
    cache.set("mat", mat)
    if task == "gen":
        return (
            html.Div(
                [
                    "File validated. Generating predictions... This may take up to 60 seconds."
                ]
            ),
            True,
            dash.no_update,
        )

    return (
        html.Div(
            [
                "File validated. Creating visualizations... This may take up to 20 seconds."
            ]
        ),
        dash.no_update,
        True,
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Output("prediction-download", "data"),
    Input("generation-ready", "data"),
    prevent_initial_call=True,
)
def generate_prediction(ready):
    filename = cache.get("filename")
    mat = cache.get("mat")
    temp_mat_path = os.path.join(TEMP_PATH, filename)
    output_path = os.path.splitext(temp_mat_path)[0] + "_prediction.mat"
    run_inference(mat, model_path=None, output_path=output_path)
    return (
        html.Div(["The prediction has been generated successfully."]),
        dcc.send_file(output_path),
    )


@app.callback(
    Output("data-upload-message", "children", allow_duplicate=True),
    Input("visualization-ready", "data"),
    prevent_initial_call=True,
)
def create_visualization(ready):
    mat = cache.get("mat")
    fig = create_fig(default_n_shown_samples=2000)
    figure = make_figure(mat)
    fig.replace(figure)
    graph.figure = fig
    return visualization_div


@app.callback(
    Output("graph", "figure"),
    Output("debug-message", "children"),
    Input("n-sample-dropdown", "value"),
    prevent_initial_call=True,
)
def change_sampling_level(sampling_level):
    sampling_level_map = {"x1": 2000, "x2": 4000, "x4": 8000}
    n_samples = sampling_level_map[sampling_level]
    mat = cache.get("mat")
    updated_pred, updated_conf = cache.get("updated_pred"), cache.get("updated_conf")
    fig = create_fig(default_n_shown_samples=n_samples)
    figure = make_figure(mat)
    figure["data"][3]["z"][0][:] = updated_pred
    figure["data"][4]["z"][0][:] = updated_pred
    figure["data"][5]["z"][0][:] = updated_pred
    figure["data"][6]["z"][0][:] = updated_conf
    fig.replace(figure)
    return fig, str(type(figure["data"][3]["z"][0]))


@app.callback(
    Output("box-select-store", "data"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("graph", "selectedData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def read_box_select(box_select, figure):
    selections = figure["layout"].get("selections")
    if not selections:
        return [], dash.no_update, dash.no_update
    if len(selections) > 1:
        selections.pop(0)
    start, end = selections[0]["x0"], selections[0]["x1"]
    start, end = round(start), round(end)
    return [start, end], figure, "Press 0 for Wake, 1 for SWS, and 2 for REM."


@app.callback(
    Output("graph", "figure", allow_duplicate=True),
    Input("box-select-store", "data"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def update_sleep_scores(box_select_range, keyboard_nevents, keyboard_event, figure):
    ctx = dash.callback_context
    if ctx.triggered:
        input_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if input_id == "keyboard":
            label = keyboard_event.get("key")
            if label in ["0", "1", "2"] and box_select_range:
                label = int(label)
                start, end = box_select_range
                annotation_history = cache.get("annotation_history")
                annotation_history.append(
                    (
                        start,
                        end + 1,
                        figure["data"][3]["z"][0][
                            start : end + 1
                        ],  # previous prediction
                        figure["data"][6]["z"][0][
                            start : end + 1
                        ],  # previous confidence
                    )
                )
                cache.set("annotation_history", annotation_history)
                figure["data"][3]["z"][0][start : end + 1] = [label] * (end - start + 1)
                figure["data"][4]["z"][0][start : end + 1] = [label] * (end - start + 1)
                figure["data"][5]["z"][0][start : end + 1] = [label] * (end - start + 1)
                figure["data"][6]["z"][0][start : end + 1] = [1] * (
                    end - start + 1
                )  # change conf to 1

                cache.set("updated_pred", figure["data"][3]["z"][0])
                cache.set("updated_conf", figure["data"][6]["z"][0])
                return figure
    return dash.no_update


@app.callback(
    Output("download-annotations", "data"),
    Input("save-button", "n_clicks"),
    State("graph", "figure"),
    prevent_initial_call=True,
)
def save_annotations(n_clicks, figure):
    mat_filename = cache.get("filename")
    temp_mat_path = os.path.join(TEMP_PATH, mat_filename)
    mat = cache.get("mat")
    mat["pred_labels"] = figure["data"][3]["z"][0]
    mat["confidence"] = figure["data"][6]["z"][0]
    savemat(temp_mat_path, mat)
    return dcc.send_file(temp_mat_path)


def open_browser():
    webbrowser.open_new(f"http://127.0.0.1:{port}/")


if __name__ == "__main__":
    run_app()
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=8050)