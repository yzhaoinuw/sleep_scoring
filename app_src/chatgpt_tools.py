# -*- coding: utf-8 -*-
"""
Placeholder helper contracts for the ChatGPT sleep-scoring workflow.

The intent is to keep model prompting focused on high-level reasoning and let
the app provide deterministic helpers for image capture, interval inspection,
and score editing.
"""


from pathlib import Path
from copy import deepcopy
from typing import Any, Literal

import numpy as np
from plotly_resampler import FigureResampler

SleepState = Literal["Wake", "NREM", "REM"]


def _merge_nested_update(target: dict[str, Any], key: str, value: Any) -> None:
    """Convert flattened Plotly update keys like marker_size into nested dicts."""
    if "_" not in key:
        target[key] = value
        return

    head, tail = key.split("_", 1)
    existing_value = target.get(head)
    if not isinstance(existing_value, dict):
        existing_value = {}
        target[head] = existing_value

    _merge_nested_update(existing_value, tail, value)


def capture_overview_snapshot(
    fig: FigureResampler,
    output_path: str | Path,
) -> Path:
    """
    Save a full-session visualization snapshot for the first coarse pass.

    Suggested implementation guidelines:
    - Use a fixed export size so prompts always see the same layout.
    - Return the final file path so the caller can attach the image to the API.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, width=1800, height=900)
    return output_path


def capture_zoom_snapshot(
    fig: FigureResampler,
    start_s: float,
    end_s: float,
    output_path: str | Path,
) -> Path:
    """
    Save a zoomed visualization for a requested interval.

    Suggested implementation guidelines:
    - Programmatically set the visible x-range before exporting.
    - Keep the subplot stack identical to the overview layout.
    - Use this only for uncertain regions, likely REM, or transition boundaries.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_fig = deepcopy(fig)

    relayout_data = {
        "xaxis4.range[0]": start_s,
        "xaxis4.range[1]": end_s,
    }

    update_data = export_fig._construct_update_data(relayout_data)

    if isinstance(update_data, list) and len(update_data) > 1:
        for trace_update in update_data[1:]:
            trace_index = trace_update["index"]
            update_payload: dict[str, Any] = {}
            for key, value in trace_update.items():
                if key == "index":
                    continue
                _merge_nested_update(update_payload, key, value)
            export_fig.data[trace_index].update(update_payload, overwrite=True)

    export_fig.update_xaxes(range=[start_s, end_s])

    export_fig.write_image(output_path, width=1800, height=900)
    return Path(output_path)


def get_interval_features(
    mat: dict[str, Any],
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """
    Return compact numeric features for one interval.

    Suggested fields:
    - delta_power_mean
    - theta_power_mean
    - theta_delta_ratio_summary
    - emg_rms / emg_burst_count
    - ne_level_mean / ne_drop_score when NE exists
    - current_score_counts or current_contiguous_blocks
    """
    del mat, start_s, end_s
    raise NotImplementedError("Implement interval feature extraction for ChatGPT refinement.")


def get_current_scores(
    scores: np.ndarray,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    """
    Return the current per-second scores for a requested interval.

    The eventual implementation should slice the current in-memory scores and
    return both the raw labels and compact contiguous blocks for prompt use.
    """
    del scores, start_s, end_s
    raise NotImplementedError("Implement score lookup for ChatGPT refinement.")


def set_scores_block(
    scores: np.ndarray,
    start_s: float,
    end_s: float,
    state: SleepState,
) -> np.ndarray:
    """
    Apply one contiguous sleep-state block to the current score array.

    Suggested behavior:
    - Round or clamp boundaries to integer-second epochs.
    - Accept `state` values Wake/NREM/REM and map them to app labels.
    - Return a new score array so undo history remains predictable.
    """
    del scores, start_s, end_s, state
    raise NotImplementedError("Implement contiguous score editing for ChatGPT output.")


def apply_transition_rules(scores: np.ndarray) -> np.ndarray:
    """
    Clean up invalid transitions after ChatGPT writes coarse blocks.

    Suggested first-pass rules:
    - no NREM immediately following REM
    - no REM immediately following Wake
    - optional minimum REM duration and brief-bout smoothing
    """
    del scores
    raise NotImplementedError("Implement transition cleanup for ChatGPT output.")


def mark_uncertain_interval(
    start_s: float,
    end_s: float,
    reason: str,
) -> dict[str, int | str]:
    """
    Record an interval that should be highlighted for manual review.

    This can eventually feed a lightweight review list or a confidence ribbon in
    the existing visualization.
    """
    return {
        "start_s": int(start_s),
        "end_s": int(end_s),
        "reason": reason,
    }
