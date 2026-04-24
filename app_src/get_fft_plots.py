# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:06:41 2025

@author: yzhao
"""

import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import hamming

from app_src.config import (
    SPECTROGRAM_COLORSCALE,
    GAUSSIAN_FILTER_SIGMA,
    THETA_DELTA_RATIO_LINE_COLOR,
    THETA_DELTA_RATIO_LINE_OPACITY,
)


def _get_centered_segment(eeg: np.ndarray, center_sample: int, nperseg: int) -> np.ndarray:
    segment = np.zeros(nperseg, dtype=float)
    segment_start = center_sample - nperseg // 2
    segment_end = segment_start + nperseg
    eeg_start = max(segment_start, 0)
    eeg_end = min(segment_end, eeg.size)

    if eeg_start >= eeg_end:
        return segment

    output_start = eeg_start - segment_start
    output_end = output_start + (eeg_end - eeg_start)
    segment[output_start:output_end] = eeg[eeg_start:eeg_end]
    return segment


def get_spectrogram(
    eeg: np.ndarray,
    eeg_frequency: float,
    window_duration: float,
    mfft: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an anchored one-sided PSD spectrogram.

    Time columns are centered on exact wall-clock intervals of
    ``window_duration / 2`` seconds. Each center is independently converted to
    the nearest EEG sample, which keeps fractional sampling-rate alignment
    error bounded instead of accumulating across the recording.
    """
    eeg_frequency = float(eeg_frequency)
    window_duration = float(window_duration)
    nperseg = round(eeg_frequency * window_duration)
    nfft = nperseg if mfft is None else int(mfft)
    if nfft < nperseg:
        raise ValueError("mfft must be greater than or equal to the window length")

    step_duration = window_duration / 2
    recording_duration = eeg.size / eeg_frequency
    num_columns = int(np.ceil(recording_duration / step_duration)) + 1
    time = np.arange(num_columns) * step_duration
    center_samples = np.round(time * eeg_frequency).astype(int)

    window = hamming(nperseg)
    frequencies = np.fft.rfftfreq(nfft, d=1 / eeg_frequency)
    freq_mask = frequencies <= 30
    frequencies = frequencies[freq_mask]
    Sx = np.empty((frequencies.size, time.size), dtype=float)

    scaling = eeg_frequency * np.sum(window**2)
    one_sided_scale = np.ones(nfft // 2 + 1)
    if nfft % 2 == 0:
        one_sided_scale[1:-1] = 2
    else:
        one_sided_scale[1:] = 2

    for column, center_sample in enumerate(center_samples):
        segment = _get_centered_segment(eeg, center_sample, nperseg)
        spectrum = np.fft.rfft(segment * window, n=nfft)
        psd = one_sided_scale * np.abs(spectrum) ** 2 / scaling
        Sx[:, column] = psd[freq_mask]

    return Sx, time, frequencies


def get_fft_plots(
    eeg: np.ndarray,
    eeg_frequency: float,
    start_time: float,
    window_duration=5,
    mfft=None,
) -> go.Heatmap:
    Sx, time, frequencies = get_spectrogram(
        eeg,
        eeg_frequency,
        window_duration,
        mfft,  # potentially can be set to power of 2 for speed up
    )
    time = time + start_time
    Sx_db = 10 * np.log10(np.maximum(Sx, np.finfo(float).tiny))
    delta_mask = np.where((frequencies > 1) & (frequencies <= 4))[0]
    theta_mask = np.where((frequencies > 4) & (frequencies <= 8))[0]
    delta_power = np.mean(Sx_db[delta_mask, :], axis=0)
    theta_power = np.mean(Sx_db[theta_mask, :], axis=0)
    theta_delta_ratio = theta_power - delta_power  # In dB (log space), division becomes subtraction
    Sx_db = gaussian_filter(Sx_db, sigma=GAUSSIAN_FILTER_SIGMA)
    spectrogram = go.Heatmap(
        x=time,
        y=frequencies,
        z=Sx_db,
        name="Spectrogram",
        hoverinfo="none",
        colorscale=SPECTROGRAM_COLORSCALE,
        showlegend=False,
        showscale=True,
    )
    theta_delta_ratio = go.Scatter(
        x=time,
        y=theta_delta_ratio,
        name="Theta/Delta",
        mode="lines",
        customdata=time / 3600,
        hovertemplate="<b>time</b>: %{customdata:.2f}h<extra></extra>",
        showlegend=False,
        line=dict(color=THETA_DELTA_RATIO_LINE_COLOR, width=1),
        opacity=THETA_DELTA_RATIO_LINE_OPACITY,
    )
    return spectrogram, theta_delta_ratio


# %%
if __name__ == "__main__":
    mat = loadmat("C:/Users/yzhao/python_projects/time_series/data/arch_392.mat")
    eeg = mat["eeg"].flatten()
    eeg_frequency = mat["eeg_frequency"].item()
    start_time = mat.get("start_time")
    if start_time is None:
        start_time = 0
    else:
        start_time = start_time.item()

    spectrogram, theta_delta_ratio = get_fft_plots(eeg, eeg_frequency, start_time)

    """
    duration = math.ceil(
        (eeg.size - 1) / eeg_frequency
    )  # need to round duration to an int for later
    eeg_end_time = duration + start_time
    # Create the time sequences
    time_eeg = np.linspace(start_time, eeg_end_time, eeg.size)
    eeg_end_time = math.ceil(eeg_end_time)
    time_eeg = np.expand_dims(np.arange(start_time + 1, eeg_end_time + 1), 0)
    """
