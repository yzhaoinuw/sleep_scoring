# -*- coding: utf-8 -*-
"""Flask endpoints for direct browser-side resampling and profiling logs."""

import json
import time

import numpy as np
from flask import abort, request
from plotly.utils import PlotlyJSONEncoder

from app_src.config import PROFILE_RESAMPLER_UPDATES
from app_src.resampling import (
    clamp_x_range_to_bounds,
    compact_resampler_patch,
    fig_x_bounds,
    get_fig_resampler,
    summarize_resampler_patch,
)
from app_src.server import app, cache


@app.server.get("/_sleep_scoring/current-file")
def current_file():
    """Report which mat file this window has open, for peer same-file checks."""
    return {"app": "sleep_scoring", "filepath": cache.get("filepath") or ""}


@app.server.get("/_sleep_scoring/resample")
def resample_graph_data():
    """Return a Plotly-resampler data patch for direct browser-side restyle."""
    request_start_time = time.perf_counter()
    try:
        x0 = float(request.args["x0"])
        x1 = float(request.args["x1"])
    except (KeyError, TypeError, ValueError):
        abort(400)

    if not np.isfinite(x0) or not np.isfinite(x1) or x0 == x1:
        abort(400)

    fig = get_fig_resampler()
    if fig is None:
        abort(404)

    x_range = clamp_x_range_to_bounds(x0, x1, fig_x_bounds(fig))
    construct_start_time = time.perf_counter()
    update_patch = fig.construct_update_data_patch(
        {
            "xaxis4.range[0]": x_range[0],
            "xaxis4.range[1]": x_range[1],
        }
    )
    construct_ms = (time.perf_counter() - construct_start_time) * 1000
    update_patch = compact_resampler_patch(update_patch)
    patch_json = (
        update_patch.to_plotly_json() if hasattr(update_patch, "to_plotly_json") else update_patch
    )
    payload_start_time = time.perf_counter()
    payload = json.dumps(patch_json, cls=PlotlyJSONEncoder, separators=(",", ":"))
    payload_ms = (time.perf_counter() - payload_start_time) * 1000

    if PROFILE_RESAMPLER_UPDATES:
        payload_size_kb = len(payload.encode("utf-8")) / 1024
        callback_total_ms = (time.perf_counter() - request_start_time) * 1000
        patch_summary = summarize_resampler_patch(update_patch, fig)
        print(
            "[resampler-direct] "
            f"construct={construct_ms:.1f} ms, "
            f"payload_encode={payload_ms:.1f} ms, "
            f"total={callback_total_ms:.1f} ms, "
            f"payload={payload_size_kb:.1f} KB, "
            f"x_width={x_range[1] - x_range[0]:.1f} s, "
            f"xaxis4.range={x_range}, "
            f"{patch_summary}",
            flush=True,
        )

    return app.server.response_class(
        payload,
        mimetype="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.server.post("/_sleep_scoring/profile-log")
def browser_profile_log():
    """Mirror browser-side response-time logs into the app terminal."""
    if not PROFILE_RESAMPLER_UPDATES:
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    event = payload.get("event", "browser")
    details = [
        f"{key}={value}" for key, value in payload.items() if key != "event" and value is not None
    ]
    print(f"[{event}] " + ", ".join(details), flush=True)
    return ("", 204)
