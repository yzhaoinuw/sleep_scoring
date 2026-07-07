# -*- coding: utf-8 -*-
"""Per-recording setup: temp-dir housekeeping, cache initialization,
metadata extraction, and figure creation.
"""

import math
from collections import deque
from pathlib import Path

from app_src.make_figure import make_figure
from app_src.resampling import clear_fig_resamplers, mat_x_bounds, store_fig_resampler
from app_src.server import TEMP_PATH


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
    cache.set("filepath", filepath)
    prev_filename = cache.get("filename")
    filename = Path(filepath).stem
    # attempt for salvaging unsaved annotations
    if prev_filename is None or prev_filename != filename:
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
