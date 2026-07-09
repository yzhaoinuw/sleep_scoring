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
from app_src.config import INSTANCE_SLOT


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


def adopt_legacy_temp_files(root, slot_dir):
    """Move loose files from the pre-multi-session flat temp dir into slot 0
    so unsaved-annotation salvage survives the upgrade. Runs only on the
    first launch after the upgrade, while slot 0's dir does not exist yet.
    """
    if slot_dir.exists() or not root.is_dir():
        return
    slot_dir.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if item.is_file():
            item.rename(slot_dir / item.name)


# Each app window is its own process on its own slot; per-slot dirs keep the
# windows' caches, temp exports, and video clips from clobbering each other.
_TEMP_ROOT = Path(tempfile.gettempdir()) / "sleep_scoring_app_data"
TEMP_PATH = _TEMP_ROOT / f"slot_{INSTANCE_SLOT}"
if INSTANCE_SLOT == 0:
    adopt_legacy_temp_files(_TEMP_ROOT, TEMP_PATH)
TEMP_PATH.mkdir(parents=True, exist_ok=True)

_VIDEO_ROOT = Path(__file__).parent / "assets" / "videos"
VIDEO_DIR = _VIDEO_ROOT / f"slot_{INSTANCE_SLOT}"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
if INSTANCE_SLOT == 0:
    for stale_clip in _VIDEO_ROOT.glob("*.mp4"):  # clips from pre-multi-session builds
        try:
            stale_clip.unlink()
        except OSError:
            pass


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
