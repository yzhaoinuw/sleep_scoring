# -*- coding: utf-8 -*-
"""Clientside (in-browser) callback registrations: mode switching, navigation,
selection, annotation, and message cleanup.

The JavaScript implementations live in ``app_src/assets/clientsideCallbacks.js``
under the ``dash_clientside.sleep_scoring`` namespace, mirroring the names and
sections here.
"""

from dash import ClientsideFunction
from dash.dependencies import Input, Output, State

from app_src.server import app


# ---- mode switching and navigation ----

# switch_mode by pressing "m"
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="switch_mode"),
    Output("graph", "figure"),
    Output("annotation-message", "children"),
    Output("video-button", "style"),
    Output("pred-button", "style"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "figure"),
)


# pan_figure
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="pan_figure"),
    Output("graph", "figure", allow_duplicate=True),
    Output("graph", "relayoutData"),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("graph", "relayoutData"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# apply_direct_restyle
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="apply_direct_restyle"),
    Output("graph-direct-restyle-status-store", "data"),
    Input("graph-direct-restyle-payload-store", "data"),
    prevent_initial_call=True,
)


# ---- selection ----

# read_box_select
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="read_box_select"),
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
    ClientsideFunction(namespace="sleep_scoring", function_name="read_click_select"),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph", "clickData"),
    State("graph", "figure"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)


# read_bout_context_select
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="read_bout_context_select"),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph-contextmenu", "n_events"),
    State("graph-contextmenu", "event"),
    State("graph", "figure"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)


# read_annotation_auto_pan_select
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="read_annotation_auto_pan_select"),
    Output("box-select-store", "data", allow_duplicate=True),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Output("video-button", "style", allow_duplicate=True),
    Input("graph-annotation-select", "n_events"),
    State("graph-annotation-select", "event"),
    State("graph", "figure"),
    State("mat-metadata-store", "data"),
    prevent_initial_call=True,
)


# ---- annotation ----

# make_annotation
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="make_annotation"),
    Output("video-button", "style", allow_duplicate=True),
    Output("updated-sleep-scores-store", "data", allow_duplicate=True),
    Output("box-select-store", "data", allow_duplicate=True),
    Input("keyboard", "n_events"),
    State("keyboard", "event"),
    State("box-select-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# update_sleep_scores
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="update_sleep_scores"),
    Output("graph", "figure", allow_duplicate=True),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("updated-sleep-scores-store", "data"),
    State("graph", "figure"),
    prevent_initial_call=True,
)


# ---- message cleanup ----

# clear_display
app.clientside_callback(
    ClientsideFunction(namespace="sleep_scoring", function_name="clear_display"),
    Output("annotation-message", "children", allow_duplicate=True),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)


# ---- debug ----

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
