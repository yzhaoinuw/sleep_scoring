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
THETA_DELTA_RATIO_LINE_COLOR = "white"  # "black", "white", etc.

# %% Automatic sleep scoring customization
POSTPROCESS = True

# %% Others
PORT = 8050
