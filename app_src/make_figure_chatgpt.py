# -*- coding: utf-8 -*-
"""ChatGPT-specific figure builder for model-facing snapshot exports."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB

from app_src.config import FIX_NE_Y_RANGE
from app_src.get_fft_plots import get_fft_plots
from app_src.make_figure_dev import (
    COLORSCALE,
    HEATMAP_WIDTH,
    OVERVIEW_XAXIS_NTICKS,
    RANGE_PADDING_PERCENT,
    RANGE_QUANTILE,
    SLEEP_SCORE_OPACITY,
    get_padded_sleep_scores,
)

SPECTROGRAM_Y_MAX_HZ = 15
SPECTROGRAM_Y_TICKVALS_HZ = list(range(0, SPECTROGRAM_Y_MAX_HZ + 1, 5))


def make_chatgpt_vision_figure(
    mat,
    plot_name="",
    default_n_shown_samples=2048,
    num_class=3,
):
    """Build the model-facing figure with only the spectrogram and NE panels."""
    eeg, ne = mat.get("eeg"), mat.get("ne")
    eeg_freq, ne_freq = mat.get("eeg_frequency"), mat.get("ne_frequency")
    start_time = mat.get("start_time")
    if mat.get("num_class") is not None:
        num_class = mat["num_class"]
    if start_time is None:
        start_time = 0

    duration = math.ceil((eeg.size - 1) / eeg_freq)
    sleep_scores = get_padded_sleep_scores(mat)
    eeg_end_time = duration + start_time
    eeg_end_time = math.ceil(eeg_end_time)

    np.place(
        sleep_scores,
        sleep_scores == -1,
        [np.nan],
    )
    if len(sleep_scores.shape) == 1:
        sleep_scores = np.expand_dims(sleep_scores, axis=0)
    sleep_scores = sleep_scores.tolist()

    fig = FigureResampler(
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "EEG Spectrogram",
                "NE",
            ),
            row_heights=[0.45, 0.55],
            specs=[
                [{"secondary_y": True, "r": -0.05}],
                [{"r": -0.05}],
            ],
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    sleep_scores = go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,
        z=sleep_scores,
        name="Sleep Scores",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,
    )

    spectrogram, theta_delta_ratio = get_fft_plots(eeg, eeg_freq, start_time)
    spectrogram.colorbar = dict(
        title="Power (dB)",
        orientation="h",
        thicknessmode="fraction",
        thickness=0.02,
        lenmode="fraction",
        len=0.18,
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=0.8,
        tickfont=dict(size=8),
    )

    ne_range = 1
    if ne is not None and ne.size > 1:
        ne_end_time = (ne.size - 1) / ne_freq + start_time
        time_ne = np.linspace(start_time, ne_end_time, ne.size)
        ne_lower_range, ne_upper_range = (
            np.nanquantile(ne, 1 - RANGE_QUANTILE),
            np.nanquantile(ne, RANGE_QUANTILE),
        )
        ne_range = max(abs(ne_lower_range), abs(ne_upper_range))
        if ne_range == 0:
            ne_range = 1
        fig.add_trace(
            go.Scattergl(
                name="NE",
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hovertemplate="<b>time</b>: %{x:.2f}" + "<br><b>y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=ne,
            row=2,
            col=1,
        )
    else:
        fig.add_annotation(
            text="NE unavailable",
            x=0.5,
            xref="paper",
            y=0.5,
            yref="y2 domain",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )

    fig.add_trace(spectrogram, secondary_y=False, row=1, col=1)
    fig.add_trace(theta_delta_ratio, secondary_y=True, row=1, col=1)
    fig.add_trace(sleep_scores, row=2, col=1)
    fig.update_layout(
        autosize=True,
        margin=dict(t=50, l=10, r=5, b=20),
        height=700,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=plot_name,
            font=dict(size=16),
            xanchor="left",
            x=0.03,
            automargin=True,
            yref="paper",
        ),
        xaxis2=dict(
            tickformat="digits",
            nticks=OVERVIEW_XAXIS_NTICKS,
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
    )

    fig.update_traces(xaxis="x2")
    fig.update_xaxes(range=[start_time, eeg_end_time], row=1, col=1)
    fig.update_xaxes(
        range=[start_time, eeg_end_time],
        row=2,
        col=1,
        title_text="<b>Time (s)</b>",
        title_standoff=10,
        ticklabelstandoff=5,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + RANGE_PADDING_PERCENT),
            ne_range * (1 + RANGE_PADDING_PERCENT),
        ],
        fixedrange=FIX_NE_Y_RANGE,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title="Frequency (Hz)",
        range=[0, SPECTROGRAM_Y_MAX_HZ],
        tickmode="array",
        tickvals=SPECTROGRAM_Y_TICKVALS_HZ,
        fixedrange=True,
        secondary_y=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title="Theta/Delta",
        overlaying="y",
        side="right",
        showgrid=False,
        fixedrange=True,
        secondary_y=True,
        row=1,
        col=1,
    )
    fig.update_annotations(font_size=14)
    fig["layout"]["annotations"][-1]["font"]["size"] = 14

    return fig


if __name__ == "__main__":
    import os

    import plotly.io as io
    from scipy.io import loadmat

    io.renderers.default = "browser"
    data_path = "../user_test_files/"
    mat_file = "35_app13_groundtruth.mat"
    mat = loadmat(os.path.join(data_path, mat_file), squeeze_me=True)

    mat_name = os.path.basename(mat_file)
    fig = make_chatgpt_vision_figure(mat, plot_name=mat_name)
    fig.show_dash(mode="external", config={"scrollZoom": True})
