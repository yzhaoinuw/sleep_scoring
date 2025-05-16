# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:36:14 2023

@author: Yue

Notes
1. A common reason that sleep scores, which are a heatmap,
   don't show up is that they have shape of (N,), instead of (1, N). The heatmap
   only works with 2d arrays.
"""

import math
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from app_src.get_fft_plots import get_fft_plots

# set up color config
SLEEP_SCORE_OPACITY = 1
STAGE_COLORS = [
    "rgb(124, 124, 251)",  # Wake,
    "rgb(251, 124, 124)",  # NREM,
    "rgb(123, 251, 123)",  # REM,
    "rgb(255, 255, 0)",  # MA yellow
]
STAGE_NAMES = ["Wake: 1", "NREM: 2", "REM: 3", "MA: 4"]
COLORSCALE = {
    3: [[0, STAGE_COLORS[0]], [0.5, STAGE_COLORS[1]], [1, STAGE_COLORS[2]]],
    4: [
        [0, STAGE_COLORS[0]],
        [1 / 3, STAGE_COLORS[1]],
        [2 / 3, STAGE_COLORS[2]],
        [1, STAGE_COLORS[3]],
    ],
}
RANGE_QUANTILE = 0.9999
HEATMAP_WIDTH = 40
RANGE_PADDING_PERCENT = 0.2


def get_padded_sleep_scores(mat) -> np.ndarray:
    """Make a sleep score array the same size as the duration."""
    eeg = mat.get("eeg")
    eeg_freq = mat.get("eeg_frequency")
    duration = math.ceil(
        (eeg.size - 1) / eeg_freq
    )  # need to round duration to an int for later
    sleep_scores = mat.get("sleep_scores", np.array([]))
    if sleep_scores.size == 0:
        # if unscored, initialize with nan
        sleep_scores = np.zeros(duration)
        sleep_scores[:] = np.nan
    else:
        # manually scored, but may contain missing scores
        sleep_scores = sleep_scores.astype(float)

        # sleep_scores need to have the length of duration. pad if necessary
        pad_len = duration - sleep_scores.size
        if pad_len > 0:
            sleep_scores = np.pad(
                sleep_scores, (0, pad_len), "constant", constant_values=np.nan
            )
    return sleep_scores


def make_figure(mat, plot_name="", default_n_shown_samples=2048, num_class=3):
    # Time span and frequencies
    eeg, emg, ne = mat.get("eeg"), mat.get("emg"), mat.get("ne")
    eeg_freq, ne_freq = mat.get("eeg_frequency"), mat.get("ne_frequency")
    start_time = mat.get("start_time")
    if mat.get("num_class") is not None:
        num_class = mat["num_class"]
    if start_time is None:
        start_time = 0

    duration = math.ceil(
        (eeg.size - 1) / eeg_freq
    )  # need to round duration to an int for later

    # scored fully or partially or unscored
    sleep_scores = get_padded_sleep_scores(mat)
    eeg_end_time = duration + start_time
    # Create the time sequences
    time_eeg = np.linspace(start_time, eeg_end_time, eeg.size)
    eeg_end_time = math.ceil(time_eeg[-1])
    eeg_lower_range, eeg_upper_range = np.nanquantile(
        eeg, 1 - RANGE_QUANTILE
    ), np.nanquantile(eeg, RANGE_QUANTILE)
    emg_lower_range, emg_upper_range = np.nanquantile(
        emg, 1 - RANGE_QUANTILE
    ), np.nanquantile(emg, RANGE_QUANTILE)
    eeg_range = max(abs(eeg_lower_range), abs(eeg_upper_range))
    emg_range = max(abs(emg_lower_range), abs(emg_upper_range))
    np.place(
        sleep_scores, sleep_scores == -1, [np.nan]
    )  # convert -1 to None for heatmap visualization

    # convert flat array to 2D array for visualization to work
    if len(sleep_scores.shape) == 1:
        sleep_scores = np.expand_dims(sleep_scores, axis=0)

    fig = FigureResampler(
        make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                "EEG Spectrogram",
                "EEG",
                "EMG",
                "NE",
            ),
            row_heights=[0.16, 0.28, 0.28, 0.28],
            specs=[
                [
                    {"secondary_y": True, "r": -0.05}
                ],  # Allow dual y-axes and reduce the padding on the right side
                [{"r": -0.05}],
                [{"r": -0.05}],
                [{"r": -0.05}],
            ],
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    # Create a heatmap for stages
    sleep_scores = go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,  # assuming that the max abs value of eeg, emg, or ne is no more than 10
        z=sleep_scores,
        name="Sleep Scores",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,  # add small gaps to serve as boundaries / ticks
    )

    spectrogram, theta_delta_ratio = get_fft_plots(eeg, eeg_freq, start_time)
    spectrogram.colorbar = dict(
        title="Power (dB)",
        orientation="h",
        thicknessmode="fraction",  # set the mode of thickness to fraction
        thickness=0.02,  # the thickness of the colorbar
        lenmode="fraction",  # set the mode of length to fraction
        len=0.15,  # the length of the colorbar
        yanchor="bottom",
        y=1,  # the y position of the colorbar
        xanchor="right",  # anchor the colorbar at the left
        x=0.8,  # the x position of the colorbar
        tickfont=dict(size=8),
    )
    # Add the time series to the figure
    fig.add_trace(
        go.Scattergl(
            name="EEG",
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
        ),
        hf_x=time_eeg,
        hf_y=eeg,
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            name="EMG",
            line=dict(width=1),
            marker=dict(size=2, color="black"),
            showlegend=False,
            mode="lines+markers",
            hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
        ),
        hf_x=time_eeg,
        hf_y=emg,
        row=3,
        col=1,
    )

    for i, color in enumerate(STAGE_COLORS[:num_class]):
        fig.add_trace(
            go.Scatter(
                x=[-100],
                y=[0.2],
                name=STAGE_NAMES[i],
                mode="markers",
                marker=dict(
                    size=8, color=color, symbol="square", opacity=SLEEP_SCORE_OPACITY
                ),
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    ne_lower_range, ne_upper_range = 0, 0
    if ne is not None and ne.size > 1:
        # ne = ne.flatten()
        # ne_freq = ne_freq.item()
        ne_end_time = (ne.size - 1) / ne_freq + start_time

        # Create the time sequences
        time_ne = np.linspace(start_time, ne_end_time, ne.size)
        # ne_end_time = math.ceil(ne_end_time)
        ne_lower_range, ne_upper_range = np.nanquantile(
            ne, 1 - RANGE_QUANTILE
        ), np.nanquantile(ne, RANGE_QUANTILE)
        fig.add_trace(
            go.Scattergl(
                name="NE",
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hovertemplate="<b>time</b>: %{x:.2f}"
                + "<br><b>y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=ne,
            row=4,
            col=1,
        )

    ne_range = max(abs(ne_lower_range), abs(ne_upper_range))

    # add the heatmap last so that their indices can be accessed using last indices
    fig.add_trace(spectrogram, secondary_y=False, row=1, col=1)
    fig.add_trace(theta_delta_ratio, secondary_y=True, row=1, col=1)
    fig.add_trace(sleep_scores, row=2, col=1)
    fig.add_trace(sleep_scores, row=3, col=1)
    fig.add_trace(sleep_scores, row=4, col=1)
    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=10, r=5, b=20),
        height=800,
        hovermode="x unified",  # gives crosshair in one subplot
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=plot_name,
            font=dict(size=16),
            xanchor="left",
            x=0.03,
            # yanchor="bottom",
            # y=0.92,
            automargin=True,
            yref="paper",
        ),
        # yaxis4=dict(tickvals=[]),  # suppress y ticks on the heatmap
        xaxis4=dict(tickformat="digits"),
        legend=dict(
            x=0.6,  # adjust these values to position the sleep score legend STAGE_NAMES
            y=0.85,
            orientation="h",  # makes legend items horizontal
            bgcolor="rgba(0,0,0,0)",  # transparent legend background
            font=dict(size=10),  # adjust legend text size
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
    )

    fig.update_traces(xaxis="x4")  # gives crosshair across all subplots
    fig.update_xaxes(range=[start_time, eeg_end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, eeg_end_time], row=2, col=1)
    fig.update_xaxes(range=[start_time, eeg_end_time], row=3, col=1)
    fig.update_xaxes(
        range=[start_time, eeg_end_time],
        row=4,
        col=1,
        title_text="<b>Time (s)</b>",
        title_standoff=10,
        ticklabelstandoff=5,  # keep some distance between tick label and the minor ticks
        minor=dict(
            tick0=0,
            dtick=3600,
            tickcolor="black",
            ticks="outside",
            ticklen=5,
            tickwidth=2,
        ),
    )
    fig.update_yaxes(
        range=[
            eeg_range * -(1 + RANGE_PADDING_PERCENT),
            eeg_range * (1 + RANGE_PADDING_PERCENT),
        ],
        # fixedrange=True,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[
            emg_range * -(1 + RANGE_PADDING_PERCENT),
            emg_range * (1 + RANGE_PADDING_PERCENT),
        ],
        # fixedrange=True,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + RANGE_PADDING_PERCENT),
            ne_range * (1 + RANGE_PADDING_PERCENT),
        ],
        fixedrange=True,
        row=4,
        col=1,
    )
    fig.update_yaxes(
        title="Frequency (Hz)",
        range=[0, 30],
        fixedrange=True,
        secondary_y=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title="Theta/Delta",
        overlaying="y",
        side="right",
        fixedrange=True,
        secondary_y=True,
        row=1,
        col=1,
    )
    fig.update_annotations(font_size=14)  # subplot title size
    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import os
    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    data_path = "../user_test_files/"
    mat_file = "35_app13.mat"
    mat = loadmat(os.path.join(data_path, mat_file), squeeze_me=True)
    # mat_file = "C:/Users/yzhao/python_projects/sleep_scoring/user_test_files/box1_COM18_RZ10_2_1_2024-06-03_09-04-56-902_sdreamer_3class.mat"
    # mat = loadmat(mat_file)
    mat_name = os.path.basename(mat_file)
    fig = make_figure(mat, plot_name=mat_name)
    fig.show_dash(config={"scrollZoom": True})
