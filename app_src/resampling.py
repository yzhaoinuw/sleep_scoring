# -*- coding: utf-8 -*-
"""Plotly-resampler figure state shared by the resample route and the
graph-navigation callbacks, plus patch and profiling helpers.
"""

import json
import math

import numpy as np
from plotly.utils import PlotlyJSONEncoder

NAVIGATION_LATEST_PROFILE_ID = 0
RESAMPLER_PATCH_X_DECIMALS = 3
RESAMPLER_PATCH_Y_DECIMALS = 7
FIG_RESAMPLER = None


def clear_fig_resamplers():
    global FIG_RESAMPLER, NAVIGATION_LATEST_PROFILE_ID

    FIG_RESAMPLER = None
    NAVIGATION_LATEST_PROFILE_ID = 0


def store_fig_resampler(fig):
    global FIG_RESAMPLER

    FIG_RESAMPLER = fig


def get_fig_resampler():
    return FIG_RESAMPLER


def mat_x_bounds(mat):
    eeg = mat.get("eeg")
    eeg_freq = mat.get("eeg_frequency")
    start_time = mat.get("start_time", 0)
    if eeg is None or not eeg_freq:
        return None

    try:
        start = float(start_time)
        end = start + math.ceil((eeg.size - 1) / float(eeg_freq))
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return None
    return [start, end]


def fig_x_bounds(fig):
    meta = getattr(getattr(fig, "layout", None), "meta", None)
    if isinstance(meta, dict):
        bounds = meta.get("sleepScoringXBounds")
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            try:
                x0 = float(bounds[0])
                x1 = float(bounds[1])
            except (TypeError, ValueError):
                return None
            if np.isfinite(x0) and np.isfinite(x1) and x1 > x0:
                return [x0, x1]
    return None


def clamp_x_range_to_bounds(x0, x1, bounds):
    low = min(x0, x1)
    high = max(x0, x1)
    if not bounds:
        return [low, high]

    bound_low, bound_high = bounds
    span = bound_high - bound_low
    width = high - low
    if width >= span:
        return [bound_low, bound_high]

    clamped_low = min(max(low, bound_low), bound_high - width)
    return [clamped_low, clamped_low + width]


def compact_resampler_patch(update_patch):
    """Trim numeric precision in resampler trace updates before Dash serializes them."""
    if not hasattr(update_patch, "_operations"):
        return update_patch

    for operation in update_patch._operations:
        location = operation.get("location", [])
        if len(location) < 3 or location[0] != "data":
            continue

        trace_property = location[2]
        if trace_property == "x":
            decimals = RESAMPLER_PATCH_X_DECIMALS
        elif trace_property == "y":
            decimals = RESAMPLER_PATCH_Y_DECIMALS
        else:
            continue

        value = operation.get("params", {}).get("value")
        if value is None:
            continue

        try:
            operation["params"]["value"] = np.round(
                np.asarray(value, dtype=float),
                decimals,
            ).tolist()
        except (TypeError, ValueError):
            continue

    return update_patch


def summarize_resampler_patch(update_patch, fig, max_items=6):
    """Return a compact size breakdown for Dash Patch trace updates."""
    if not hasattr(update_patch, "to_plotly_json"):
        return "operations=n/a"

    operations = update_patch.to_plotly_json().get("operations", [])
    items = []
    total_operation_size = 0
    for operation in operations:
        location = operation.get("location", [])
        if len(location) < 3 or location[0] != "data":
            continue

        trace_index = location[1]
        trace_property = location[2]
        value = operation.get("params", {}).get("value")
        value_json = json.dumps(value, cls=PlotlyJSONEncoder, separators=(",", ":"))
        value_size_kb = len(value_json.encode("utf-8")) / 1024
        total_operation_size += value_size_kb

        trace_name = ""
        try:
            trace_name = fig.data[trace_index].name or ""
        except (IndexError, TypeError, AttributeError):
            pass
        trace_label = f"{trace_index}:{trace_name or trace_property}"
        items.append((value_size_kb, trace_label, trace_property))

    items.sort(reverse=True)
    item_text = ", ".join(
        f"{trace_label}.{trace_property}={value_size_kb:.1f} KB"
        for value_size_kb, trace_label, trace_property in items[:max_items]
    )
    return (
        f"operations={len(operations)}, "
        f"operation_values={total_operation_size:.1f} KB, "
        f"top=[{item_text}]"
    )


def build_direct_restyle_payload(update_patch, browser_profile_marker):
    """Serialize resampler Patch operations for browser-side Plotly.restyle."""
    if not hasattr(update_patch, "to_plotly_json"):
        return update_patch

    return {
        "applyPath": "direct-restyle",
        "profileMarker": browser_profile_marker,
        "operations": update_patch.to_plotly_json().get("operations", []),
    }


def relayout_event_to_data(relayout_event):
    if not relayout_event:
        return None

    x0 = relayout_event.get("detail.x0")
    x1 = relayout_event.get("detail.x1")
    if x0 is None or x1 is None:
        return None

    try:
        x0 = float(x0)
        x1 = float(x1)
    except (TypeError, ValueError):
        return None

    return {
        "xaxis4.range[0]": min(x0, x1),
        "xaxis4.range[1]": max(x0, x1),
    }


def relayout_event_to_mode(relayout_event):
    if not relayout_event:
        return "final"

    mode = relayout_event.get("detail.mode")
    if mode == "fast":
        return "fast"

    return "final"


def relayout_event_to_profile_marker(relayout_event):
    if not relayout_event:
        return None

    profile_id = relayout_event.get("detail.profileId")
    if profile_id is None:
        return None

    try:
        profile_id = int(profile_id)
    except (TypeError, ValueError):
        return None

    return {
        "profileId": profile_id,
        "mode": relayout_event_to_mode(relayout_event),
        "source": relayout_event.get("detail.source") or "",
    }


def format_profile_ms(value):
    if value is None:
        return "n/a"

    try:
        return f"{float(value):.1f} ms"
    except (TypeError, ValueError):
        return "n/a"


def navigation_profile_id(profile_marker):
    if not profile_marker:
        return None
    return profile_marker.get("profileId")


def mark_navigation_profile_seen(profile_id):
    global NAVIGATION_LATEST_PROFILE_ID

    if profile_id is None:
        return False

    if profile_id < NAVIGATION_LATEST_PROFILE_ID:
        return True

    if profile_id > NAVIGATION_LATEST_PROFILE_ID:
        NAVIGATION_LATEST_PROFILE_ID = profile_id

    return False


def is_stale_navigation_profile(profile_id):
    return profile_id is not None and profile_id < NAVIGATION_LATEST_PROFILE_ID


def latest_navigation_profile_id():
    return NAVIGATION_LATEST_PROFILE_ID
