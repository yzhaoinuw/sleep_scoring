# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:27:03 2023

@author: yzhao
"""

# import dash_uploader as du
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_extensions import EventListener


# %% home div, aka inital layout

upload_button_style = {
    "fontSize": "18px",
    "width": "15%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    "marginLeft": "15px",
    "marginTop": "15px",
    # "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
    "padding": "0px",
}

mat_upload_button = html.Button(
    "Click here to select a mat file",
    id="mat-upload-button",
    style=upload_button_style,
)

backend_div = html.Div(
    children=[
        dcc.Store(id="mat-metadata-store"),
        dcc.Store(id="prediction-ready-store"),
        dcc.Store(id="visualization-ready-store"),
        dcc.Store(id="box-select-store"),
        dcc.Store(id="update-fft-store"),
        dcc.Store(id="video-path-store"),
        dcc.Store(id="clip-name-store"),
        dcc.Store(id="clip-range-store"),
        dcc.Store(id="updated-sleep-scores-store"),
        dcc.Store(id="backup-sleep-scores-store"),
        EventListener(
            id="keyboard",
            events=[{"event": "keydown", "props": ["key"]}],
        ),
        dcc.Interval(
            id="interval-component",
            interval=1 * 1000,  # in milliseconds
            max_intervals=0,  # stop after the first interval
        ),
    ]
)

home_div = html.Div(
    [
        mat_upload_button,
        html.Div(id="data-upload-message", style={"marginLeft": "10px"}),
        # html.Div(id="annotation-message", style={"marginLeft": "10px"}),
        html.Div(id="debug-message", style={"marginLeft": "10px"}),
        backend_div,
        # dcc.Store(id="mat-metadata-store"),
        # dcc.Store(id="prediction-ready-store"),
        # dcc.Store(id="visualization-ready-store"),
        # dcc.Store(id="net-annotation-count-store"),
        # dcc.Download(id="prediction-download-store"),
        # pred_modal_confirm,
    ]
)

# %% Dynamic components, ie., the components below appear as results of callbacks,
# when used as Input for callbacks, need to guard against automatic firing,
# prevent_inital_call won't work

video_upload_box_style = {
    "fontSize": "18px",
    "width": "100%",
    "height": "auto",
    "minHeight": "auto",
    "lineHeight": "auto",
    "borderWidth": "1px",
    "borderStyle": "none",
    "textAlign": "center",
    "margin": "5px",  # spacing between the upload box and the div it's in
    "borderRadius": "10px",  # rounded corner
    "backgroundColor": "lightgrey",
}

video_upload_button = html.Button(
    "Click here to select a video file",
    id="video-upload-button",
    style=video_upload_box_style,
)


pred_modal_confirm = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Prediction")),
        dbc.ModalBody(
            "Generating predictions will overwrite all the current sleep scores. Are you sure?"
        ),
        dbc.ModalFooter(html.Button("Yes", id="pred-confirm-button")),
    ],
    id="pred-modal-confirm",
    size="lg",
    is_open=False,
    # backdrop="static",  # the user must clicks the "x" to exit
    centered=True,
)

save_div = html.Div(
    style={
        "display": "flex",
        "marginRight": "10px",
        "marginLeft": "10px",
        "marginBottom": "10px",
    },
    children=[
        html.Button(
            "Save Annotations",
            id="save-button",
            # style={"visibility": "visible"},
        ),
        dcc.Download(id="download-annotations"),
        dcc.Download(id="download-spreadsheet"),
        html.Button(
            "Undo Annotation",
            id="undo-button",
            style={"visibility": "hidden"},
        ),
    ],
)


graph = dcc.Graph(
    id="graph",
    config={
        "scrollZoom": True,
    },
)

video_modal = dbc.Modal(
    [
        # dbc.ModalHeader(dbc.ModalTitle("Video")),
        dbc.ModalHeader(html.Div(id="video-title")),
        dbc.ModalBody(html.Div(id="video-container")),
        dbc.ModalFooter(html.Div(id="video-message")),
    ],
    id="video-modal",
    size="lg",
    is_open=False,
    scrollable=True,
    backdrop="static",  # the user must clicks the "x" to exit
    centered=True,
)

reselect_video_button = html.Button(
    "Select a different video", id="reselect-video-button"
)


# %%
def make_utility_div(pred_disabled=True):
    # enable or disable pred button depending on availability of pytorch
    pred_button = html.Button(
        "Generate Predictions",
        id="pred-button",
        style={"visibility": "hidden"},
    )
    if pred_disabled:
        pred_button = html.Button(
            "Generate Predictions",
            id="pred-button",
            style={"visibility": "hidden"},
            disabled=True,
            title="Add Torch to generate predictions.",
        )
    utility_div = html.Div(
        style={
            "display": "flex",
            "marginLeft": "10px",
            "marginTop": "5px",
            "marginBottom": "0px",
            "justifyContent": "flex-start",
            "width": "100%",
            "alignItems": "center",
            "flexWrap": "nowrap",  # prevent wrap during transition
            "whiteSpace": "nowrap",
            "paddingRight": "30px",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={"display": "flex", "marginLeft": "10px", "gap": "10px"},
                children=[
                    html.Div(["Sampling Level"]),
                    dcc.Dropdown(
                        options=["x1", "x2", "x4"],
                        value="x1",
                        id="n-sample-dropdown",
                        searchable=False,
                        clearable=False,
                    ),
                    html.Div(
                        [
                            html.Button(
                                "Check Video",
                                id="video-button",
                                style={"visibility": "hidden"},
                            )
                        ]
                    ),
                ],
            ),
            html.Div(
                [pred_button],
                style={"marginLeft": "auto"},  # keep the button to the right edge
            ),
        ],
    )
    return utility_div


def make_visualization_div(pred_disabled=True):
    utility_div = make_utility_div(pred_disabled)
    visualization_div = html.Div(
        children=[
            utility_div,
            video_modal,
            pred_modal_confirm,
            html.Div(
                children=[graph],
                style={"marginTop": "1px", "marginLeft": "20px", "marginRight": "15px"},
            ),
            # backend_div,
            html.Div(
                id="annotation-message",
                style={
                    "marginLeft": "10px",
                    "minHeight": "18px",
                },
            ),
            save_div,
        ],
    )
    return visualization_div


# %%
class Components:
    def __init__(self, pred_disabled=True):
        self.home_div = home_div
        self.graph = graph
        self.visualization_div = make_visualization_div(pred_disabled)
        self.mat_upload_button = mat_upload_button
        self.video_upload_button = video_upload_button
        self.reselect_video_button = reselect_video_button
