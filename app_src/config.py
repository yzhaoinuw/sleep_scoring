# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:55:48 2025

@author: yzhao
"""

# %% Default app window size
WINDOW_CONFIG = {
    "width": 1600,
    "height": 1000,
    "min_size": (1200, 800),
    "resizable": True,
}

# %% Figure customization
FIX_NE_Y_RANGE = False  # True or False

# see https://plotly.com/python/builtin-colorscales/, under Section Built-In Sequential Color scales
SPECTROGRAM_COLORSCALE = "viridis"  # "turbo", "jet", etc.

# consider to change this too if you change the spectrogram colorscale above
THETA_DELTA_RATIO_LINE_COLOR = "black"  # "black", "white", etc.
THETA_DELTA_RATIO_OPACITY = 0.2

GAUSSIAN_FILTER_SIGMA = 4  # how agressive to smooth the spectrogram or theta/delta line

# %% Automatic sleep scoring customization
POSTPROCESS = True
SLEEP_SCORING_MODEL = "sdreamer"  # "sdreamer" or "stats_model"

# Statistical Wake/REM model user-facing tuning
STATS_MODEL_WAKE_THRESHOLD = 0.7 # turn up in 0.05 increments to label Wake more aggresively. Range: 0 - 1.
STATS_MODEL_MIN_WAKE_DURATION = 5.0 # minimum Wake duration in seconds
STATS_MODEL_MIN_REM_DURATION = 30.0 # minimum REM duration in seconds

# %% Others
PORT = 8050
