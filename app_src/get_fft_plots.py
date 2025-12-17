# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:06:41 2025

@author: yzhao
"""

import numpy as np

import plotly.graph_objects as go

from scipy.io import loadmat
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from app_src.config import SPECTROGRAM_COLORSCALE, THETA_DELTA_RATIO_LINE_COLOR


def get_fft_plots(
    eeg: np.ndarray,
    eeg_frequency: float,
    start_time: float,
    window_duration=5,
    mfft=None,
) -> go.Heatmap:

    nperseg = round(eeg_frequency * window_duration)
    noverlap = round(nperseg / 2)
    window = hamming(nperseg)
    SFT = ShortTimeFFT(
        window,
        hop=noverlap,
        fs=eeg_frequency,
        fft_mode="onesided",
        mfft=mfft,  # potentially can be set to power of 2 for speed up
        scale_to="psd",
    )
    Sx = SFT.spectrogram(eeg)
    time = SFT.t(len(eeg)) + start_time
    frequencies = SFT.f
    freq_mask = frequencies <= 30
    frequencies = frequencies[freq_mask]
    Sx = Sx[freq_mask, :]
    Sx_db = 10 * np.log10(Sx)
    delta_mask = np.where((frequencies > 1) & (frequencies <= 4))[0]
    theta_mask = np.where((frequencies > 4) & (frequencies <= 8))[0]
    delta_power = np.mean(Sx_db[delta_mask, :], axis=0)
    theta_power = np.mean(Sx_db[theta_mask, :], axis=0)
    theta_delta_ratio = (
        delta_power / theta_power
    )  # flip delta and theta because their "magnitude" is negative
    theta_delta_ratio = gaussian_filter1d(theta_delta_ratio, 4)
    Sx_db = gaussian_filter(Sx_db, sigma=4)
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
        opacity=0.4,
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
