# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:55:48 2025

@author: yzhao
"""

import os

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
THETA_DELTA_RATIO_LINE_OPACITY = 0.2

GAUSSIAN_FILTER_SIGMA = 4  # how agressive to smooth the spectrogram or theta/delta line

# %% Automatic sleep scoring customization
POSTPROCESS = True
SLEEP_SCORING_MODEL = "stats_model"  # "sdreamer" or "stats_model"

# Statistical Wake/REM model user-facing tuning
STATS_MODEL_WAKE_THRESHOLD = (
    0.7  # turn up in 0.05 increments to label Wake more aggresively. Range: 0 - 1.
)
STATS_MODEL_MIN_WAKE_DURATION = 5.0  # minimum Wake duration in seconds
STATS_MODEL_MIN_REM_DURATION = 30.0  # minimum REM duration in seconds

# %% Multi-session
# run_desktop_app.py claims a window slot (one port per app window) and
# exports these env vars before app_src is imported. Direct imports (tests,
# scripts, --smoke) see no env vars and default to slot 0 with no peers,
# which is exactly the single-window behavior.


def _read_instance_slot():
    try:
        return max(0, int(os.environ.get("SLEEP_SCORING_INSTANCE_SLOT", "0")))
    except ValueError:
        return 0


def _read_peer_ports():
    ports = []
    for token in os.environ.get("SLEEP_SCORING_PEER_PORTS", "").split(","):
        token = token.strip()
        if token.isdigit():
            ports.append(int(token))
    return ports


INSTANCE_SLOT = _read_instance_slot()
PEER_PORTS = _read_peer_ports()

# %% Profiling
# Off by default for shipped users. Override per-run with one of the
# SLEEP_SCORING_* env vars below.
ENABLE_RESAMPLER_PERF_LOG = False
ENABLE_BROWSER_NAVIGATION_PERF_LOG = False

# %% Navigation performance
ENABLE_DIRECT_PLOTLY_RESTYLE = True

RESAMPLER_PERF_LOG = os.environ.get("SLEEP_SCORING_PROFILE_RESAMPLER", "1") != "0" and (
    ENABLE_RESAMPLER_PERF_LOG
    or os.environ.get("SLEEP_SCORING_RESAMPLER_PERF_LOG", "0") == "1"
    or os.environ.get("SLEEP_SCORING_PROFILE_RESAMPLER", "0") == "1"
)
BROWSER_NAVIGATION_PERF_LOG = os.environ.get("SLEEP_SCORING_BROWSER_NAV_PERF_LOG", "1") != "0" and (
    ENABLE_BROWSER_NAVIGATION_PERF_LOG
    or os.environ.get("SLEEP_SCORING_BROWSER_NAV_PERF_LOG", "0") == "1"
    or RESAMPLER_PERF_LOG
)
PROFILE_RESAMPLER_UPDATES = RESAMPLER_PERF_LOG

if INSTANCE_SLOT > 0:
    # Perf logging stays on the first window only; later windows would
    # interleave their output with the first window's stream.
    RESAMPLER_PERF_LOG = False
    BROWSER_NAVIGATION_PERF_LOG = False
    PROFILE_RESAMPLER_UPDATES = False
