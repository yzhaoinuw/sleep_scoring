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

GAUSSIAN_FILTER_SIGMA = 4  # how agressive to smooth the spectrogram or theta/delta line

# %% Automatic sleep scoring customization
POSTPROCESS = True
CHATGPT_MODEL = "gpt-5.4-mini"
CHATGPT_REASONING_EFFORT = "high"
CHATGPT_CONFIDENCE_THRESHOLD = 0.5
CHATGPT_SHOW_THOUGHTS = True
CHATGPT_REFINEMENT_MODE = "fixed_sections"  # "none", "adaptive", or "fixed_sections"
CHATGPT_FIXED_REFINEMENT_SECTION_COUNT = 4
CHATGPT_USE_REFERENCE_EXAMPLES = True

# %% Others
PORT = 8050
