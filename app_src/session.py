# -*- coding: utf-8 -*-
"""Per-recording setup: temp-dir housekeeping, cache initialization,
metadata extraction, and figure creation.
"""

import json
import math
import os
from collections import deque
from pathlib import Path
from urllib.request import urlopen

from app_src.config import PEER_PORTS
from app_src.make_figure import make_figure
from app_src.resampling import clear_fig_resamplers, mat_x_bounds, store_fig_resampler
from app_src.server import TEMP_PATH


PEER_QUERY_TIMEOUT_SECONDS = 0.5

# The file open in this window, reported by the peer current-file endpoint.
# Process state, not the filesystem cache: cache entries persist across
# restarts of a slot, and a stale filepath would make peers refuse a file
# that is no longer open anywhere.
_current_filepath = None


def set_current_filepath(filepath):
    global _current_filepath
    _current_filepath = filepath


def get_current_filepath():
    return _current_filepath


def _normalize_mat_path(filepath):
    return os.path.normcase(os.path.normpath(os.path.abspath(filepath)))


def find_peer_session_with_file(filepath):
    """Return the port of a live app window that already has filepath open.

    Dead windows stop answering their port, so a crashed window's claim on a
    file evaporates with it; no lock files are involved. Anything else bound
    to a peer port is ignored unless it identifies as this app.
    """
    target = _normalize_mat_path(filepath)
    for port in PEER_PORTS:
        url = f"http://127.0.0.1:{port}/_sleep_scoring/current-file"
        try:
            with urlopen(url, timeout=PEER_QUERY_TIMEOUT_SECONDS) as response:
                payload = json.load(response)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) or payload.get("app") != "sleep_scoring":
            continue
        peer_file = payload.get("filepath")
        if peer_file and _normalize_mat_path(peer_file) == target:
            return port
    return None


def create_fig(mat, filename, default_n_shown_samples=2048):
    fig = make_figure(mat, filename, default_n_shown_samples)
    bounds = mat_x_bounds(mat)
    if bounds is not None:
        meta = fig.layout.meta if isinstance(fig.layout.meta, dict) else {}
        fig.update_layout(meta={**meta, "sleepScoringXBounds": bounds})

    store_fig_resampler(fig)
    return fig


def clear_temp_dir(filename):
    """clear mat and xlsx files written in temp"""
    for temp_file in TEMP_PATH.iterdir():
        if temp_file.suffix in [".mat", ".xlsx"]:
            if temp_file.stem == filename:
                continue
            temp_file.unlink()


def coerce_video_start_time(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0

    if not math.isfinite(value):
        return 0

    return value


def write_metadata(mat):
    eeg = mat.get("eeg")
    start_time = mat.get("start_time", 0)
    eeg_freq = mat.get("eeg_frequency")
    duration = math.ceil((eeg.size - 1) / eeg_freq)  # need to round duration to an int for later
    end_time = duration + start_time
    video_start_time = coerce_video_start_time(mat.get("video_start_time", 0))
    video_path = mat.get("video_path", "")

    if not isinstance(video_path, str):
        video_path = ""

    metadata = dict(
        [
            ("start_time", start_time),
            ("end_time", end_time),
            ("video_start_time", video_start_time),
            ("video_path", ""),
        ]
    )
    return metadata


def initialize_cache(cache, filepath):
    set_current_filepath(filepath)
    previous_filepath = cache.get("filepath")
    cache.set("filepath", filepath)
    filename = Path(filepath).stem
    # Salvage unsaved annotations only when the previous process had the same
    # recording open. Older caches may not have a filepath; resetting their
    # history is safer than matching an unrelated file by basename.
    try:
        is_same_file = previous_filepath is not None and _normalize_mat_path(
            previous_filepath
        ) == _normalize_mat_path(filepath)
    except (TypeError, ValueError, OSError):
        is_same_file = False
    if not is_same_file:
        cache.set("sleep_scores_history", deque(maxlen=2))

    clear_temp_dir(filename)
    cache.set("filename", filename)
    recent_files_with_video = cache.get("recent_files_with_video")
    if recent_files_with_video is None:
        recent_files_with_video = []
    file_video_record = cache.get("file_video_record")
    if file_video_record is None:
        file_video_record = {}
    cache.set("recent_files_with_video", recent_files_with_video)
    cache.set("file_video_record", file_video_record)
    clear_fig_resamplers()
