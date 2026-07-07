# -*- coding: utf-8 -*-
"""Dash application instance, shared cache, components, and runtime paths.

Importing this module creates the app; route and callback registration live
in app_src.routes and app_src.callbacks.
"""

import tempfile
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Dash
from flask_caching import Cache

from app_src import VERSION
from app_src.components import Components


try:
    from app_src.inference import run_inference

    components = Components(pred_disabled=False)
except ImportError:
    run_inference = None
    components = Components()


app = Dash(
    __name__,
    title=f"Sleep Scoring App {VERSION}",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],  # need this for the modal to work properly
)
app.layout = components.home_div


# debug_counter = Debug_Counter()
TEMP_PATH = Path(tempfile.gettempdir()) / "sleep_scoring_app_data"
TEMP_PATH.mkdir(parents=True, exist_ok=True)
VIDEO_DIR = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


# Note: np.nan is converted to None when reading from cache
cache = Cache(
    app.server,
    config={
        "CACHE_TYPE": "filesystem",
        "CACHE_DIR": TEMP_PATH,
        "CACHE_THRESHOLD": 30,
        "CACHE_DEFAULT_TIMEOUT": 20
        * 24
        * 3600,  # to save cache for 20 days, otherwise it is default to 300 seconds
    },
)
