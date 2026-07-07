# -*- coding: utf-8 -*-
"""Serverside callbacks for graph navigation: resampler updates on relayout
and browser navigation profiling.
"""

import json
import time

import dash
from dash.dependencies import Input, Output, State
from plotly.utils import PlotlyJSONEncoder

from app_src.config import (
    BROWSER_NAVIGATION_PERF_LOG,
    ENABLE_DIRECT_PLOTLY_RESTYLE,
    RESAMPLER_PERF_LOG,
)
from app_src.resampling import (
    build_direct_restyle_payload,
    compact_resampler_patch,
    format_profile_ms,
    get_fig_resampler,
    is_stale_navigation_profile,
    latest_navigation_profile_id,
    mark_navigation_profile_seen,
    navigation_profile_id,
    relayout_event_to_data,
    relayout_event_to_mode,
    relayout_event_to_profile_marker,
    summarize_resampler_patch,
)
from app_src.server import app


RESAMPLER_PROFILE_UPDATE_ID = 0
RESAMPLER_PROFILE_LAST_START = None
RESAMPLER_PROFILE_LAST_FINISH = None
RESAMPLER_PROFILE_ACTIVE_COUNT = 0


RESAMPLER_CALLBACK_OUTPUT = (
    Output("graph-direct-restyle-payload-store", "data")
    if ENABLE_DIRECT_PLOTLY_RESTYLE
    else Output("graph", "figure", allow_duplicate=True)
)


@app.callback(
    RESAMPLER_CALLBACK_OUTPUT,
    # Output("debug-message", "children"),
    Input("graph-relayout-coalesced", "n_events"),
    State("graph-relayout-coalesced", "event"),
    prevent_initial_call=True,
)
def update_fig_resampler(_relayout_n_events, relayout_event):
    global RESAMPLER_PROFILE_ACTIVE_COUNT
    global RESAMPLER_PROFILE_LAST_FINISH
    global RESAMPLER_PROFILE_LAST_START
    global RESAMPLER_PROFILE_UPDATE_ID

    relayoutdata = relayout_event_to_data(relayout_event)
    if relayoutdata is None:
        return dash.no_update

    if "xaxis4.range[0]" not in relayoutdata and "xaxis4.range" not in relayoutdata:
        return dash.no_update

    update_mode = relayout_event_to_mode(relayout_event)
    browser_profile_marker = relayout_event_to_profile_marker(relayout_event)
    browser_profile_id = navigation_profile_id(browser_profile_marker)

    if mark_navigation_profile_seen(browser_profile_id):
        if RESAMPLER_PERF_LOG:
            print(
                "[resampler-stale] "
                f"browser_profile_id={browser_profile_id}, "
                f"latest_browser_profile_id={latest_navigation_profile_id()}, "
                f"mode={update_mode}, phase=start",
                flush=True,
            )
        return dash.no_update

    callback_start_time = time.perf_counter()
    resampler_get_start_time = time.perf_counter()
    fig = get_fig_resampler()
    resampler_get_ms = (time.perf_counter() - resampler_get_start_time) * 1000
    if fig is None:
        return dash.no_update

    # debug_counter.increment()
    profile_id = None
    active_at_start = None
    since_prev_start_ms = None
    since_prev_finish_ms = None

    if RESAMPLER_PERF_LOG:
        RESAMPLER_PROFILE_UPDATE_ID += 1
        profile_id = RESAMPLER_PROFILE_UPDATE_ID
        active_at_start = RESAMPLER_PROFILE_ACTIVE_COUNT + 1
        RESAMPLER_PROFILE_ACTIVE_COUNT = active_at_start

        if RESAMPLER_PROFILE_LAST_START is not None:
            since_prev_start_ms = (callback_start_time - RESAMPLER_PROFILE_LAST_START) * 1000
        if RESAMPLER_PROFILE_LAST_FINISH is not None:
            since_prev_finish_ms = (callback_start_time - RESAMPLER_PROFILE_LAST_FINISH) * 1000
        RESAMPLER_PROFILE_LAST_START = callback_start_time

    try:
        construct_start_time = time.perf_counter()
        update_patch = fig.construct_update_data_patch(relayoutdata)
        construct_ms = (time.perf_counter() - construct_start_time) * 1000
        if is_stale_navigation_profile(browser_profile_id):
            if RESAMPLER_PERF_LOG:
                print(
                    "[resampler-stale] "
                    f"browser_profile_id={browser_profile_id}, "
                    f"latest_browser_profile_id={latest_navigation_profile_id()}, "
                    f"mode={update_mode}, phase=after_construct",
                    flush=True,
                )
            return dash.no_update

        update_patch = compact_resampler_patch(update_patch)

        if BROWSER_NAVIGATION_PERF_LOG and browser_profile_marker is not None:
            update_patch["layout"]["meta"]["sleepScoringNavigationProfile"] = browser_profile_marker

        callback_payload = (
            build_direct_restyle_payload(update_patch, browser_profile_marker)
            if ENABLE_DIRECT_PLOTLY_RESTYLE
            else update_patch
        )
        apply_path = "direct-restyle" if ENABLE_DIRECT_PLOTLY_RESTYLE else "dash-figure-patch"

        if RESAMPLER_PERF_LOG:
            payload_start_time = time.perf_counter()
            try:
                payload_json = json.dumps(
                    callback_payload,
                    cls=PlotlyJSONEncoder,
                    separators=(",", ":"),
                )
                payload_size_kb = len(payload_json.encode("utf-8")) / 1024
            except TypeError:
                payload_json = json.dumps(callback_payload, default=str, separators=(",", ":"))
                payload_size_kb = len(payload_json.encode("utf-8")) / 1024
            payload_ms = (time.perf_counter() - payload_start_time) * 1000
            patch_summary = summarize_resampler_patch(update_patch, fig)

            x_range = relayoutdata.get("xaxis4.range")
            if x_range is None:
                x_range = [
                    relayoutdata.get("xaxis4.range[0]"),
                    relayoutdata.get("xaxis4.range[1]"),
                ]

            x_width = None
            if (
                isinstance(x_range, list)
                and len(x_range) == 2
                and x_range[0] is not None
                and x_range[1] is not None
            ):
                x_width = x_range[1] - x_range[0]

            callback_total_ms = (time.perf_counter() - callback_start_time) * 1000
            since_prev_start_text = (
                "n/a" if since_prev_start_ms is None else f"{since_prev_start_ms:.1f} ms"
            )
            since_prev_finish_text = (
                "n/a" if since_prev_finish_ms is None else f"{since_prev_finish_ms:.1f} ms"
            )
            x_width_text = "n/a" if x_width is None else f"{x_width:.1f} s"
            browser_profile_text = browser_profile_id if browser_profile_id is not None else "n/a"

            print(
                "[resampler] "
                f"id={profile_id}, "
                f"browser_profile_id={browser_profile_text}, "
                f"active_at_start={active_at_start}, "
                f"mode={update_mode}, "
                "cache_key=fig_resampler, "
                f"since_prev_start={since_prev_start_text}, "
                f"since_prev_finish={since_prev_finish_text}, "
                f"resampler_get={resampler_get_ms:.1f} ms, "
                f"construct={construct_ms:.1f} ms, "
                f"payload_encode={payload_ms:.1f} ms, "
                f"total={callback_total_ms:.1f} ms, "
                f"payload={payload_size_kb:.1f} KB, "
                f"apply_path={apply_path}, "
                f"x_width={x_width_text}, "
                f"xaxis4.range={x_range}, "
                f"{patch_summary}",
                flush=True,
            )

        return callback_payload
    finally:
        if RESAMPLER_PERF_LOG:
            RESAMPLER_PROFILE_LAST_FINISH = time.perf_counter()
            RESAMPLER_PROFILE_ACTIVE_COUNT = max(0, RESAMPLER_PROFILE_ACTIVE_COUNT - 1)


@app.callback(
    Output("navigation-profile-store", "data"),
    Input("graph-navigation-profile", "n_events"),
    State("graph-navigation-profile", "event"),
    prevent_initial_call=True,
)
def log_browser_navigation_profile(_n_events, profile_event):
    if not BROWSER_NAVIGATION_PERF_LOG or not profile_event:
        return dash.no_update

    profile_id = profile_event.get("detail.profileId", "n/a")
    mode = profile_event.get("detail.mode", "n/a")
    source = profile_event.get("detail.source", "n/a")
    x0 = profile_event.get("detail.x0")
    x1 = profile_event.get("detail.x1")
    x_width_text = "n/a"
    try:
        x_width_text = f"{float(x1) - float(x0):.1f} s"
    except (TypeError, ValueError):
        pass

    print(
        "[browser-nav] "
        f"profile_id={profile_id}, "
        f"mode={mode}, "
        f"source={source}, "
        f"coalesce={format_profile_ms(profile_event.get('detail.coalesceMs'))}, "
        f"dash_apply={format_profile_ms(profile_event.get('detail.dashApplyMs'))}, "
        f"browser_total={format_profile_ms(profile_event.get('detail.browserTotalMs'))}, "
        f"frame_gap={format_profile_ms(profile_event.get('detail.frameGapMs'))}, "
        f"x_width={x_width_text}",
        flush=True,
    )
    return profile_event
