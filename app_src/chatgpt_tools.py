# -*- coding: utf-8 -*-
"""
Helper contracts for the ChatGPT sleep-scoring workflow.

The intent is to keep model prompting focused on high-level reasoning and let
the app provide deterministic helpers for image capture, interval inspection,
and score editing.
"""


import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
from plotly_resampler import FigureResampler

from app_src.get_fft_plots import get_fft_plots
from app_src.make_figure_dev import get_padded_sleep_scores

SleepState = Literal["Wake", "NREM", "REM"]
SLEEP_STAGE_LOOKUP = {
    0: "Wake",
    1: "NREM",
    2: "REM",
}
SLEEP_STAGE_TO_SCORE = {label: score for score, label in SLEEP_STAGE_LOOKUP.items()}


def _scalar_value(value: Any, default: float = 0.0) -> float:
    """Return a float for MATLAB-loaded scalar values or a default."""
    if value is None:
        return float(default)

    array = np.asarray(value)
    if array.size == 0:
        return float(default)

    return float(array.reshape(-1)[0])


def _as_1d_float_array(value: Any) -> np.ndarray:
    """Normalize an input into a flat float array."""
    if value is None:
        return np.array([], dtype=float)

    return np.asarray(value, dtype=float).reshape(-1)


def _optional_float(value: Any) -> float | None:
    """Convert finite numeric values to Python floats and map invalid values to None."""
    if value is None:
        return None

    numeric = float(value)
    if not np.isfinite(numeric):
        return None

    return numeric


def _find_trace(
    fig: FigureResampler | None,
    trace_name: str,
    fallback_index: int,
):
    """Find a trace by name, falling back to a known relative position."""
    if fig is None:
        return None

    for trace in reversed(fig.data):
        if getattr(trace, "name", None) == trace_name:
            return trace

    if len(fig.data) >= abs(fallback_index):
        return fig.data[fallback_index]

    return None


def _get_recording_bounds(mat: dict[str, Any]) -> tuple[float, float]:
    """Return the visible recording time bounds in seconds."""
    start_time = _scalar_value(mat.get("start_time"), default=0.0)
    duration = float(get_padded_sleep_scores(mat).size)
    return start_time, start_time + duration


def _clamp_interval(
    mat: dict[str, Any],
    start_s: float,
    end_s: float,
) -> tuple[float, float]:
    """Clamp an interval to the recording range and validate it."""
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s.")

    recording_start, recording_end = _get_recording_bounds(mat)
    interval_start = max(float(start_s), recording_start)
    interval_end = min(float(end_s), recording_end)

    if interval_end <= interval_start:
        raise ValueError("Requested interval falls outside the recording.")

    return interval_start, interval_end


def _slice_signal(
    signal: np.ndarray,
    frequency_hz: float,
    start_s: float,
    end_s: float,
    recording_start_s: float,
) -> np.ndarray:
    """Slice a continuous signal using half-open [start_s, end_s) semantics."""
    if signal.size == 0 or frequency_hz <= 0:
        return np.array([], dtype=float)

    start_idx = int(np.floor((start_s - recording_start_s) * frequency_hz))
    end_idx = int(np.ceil((end_s - recording_start_s) * frequency_hz))

    start_idx = max(0, min(start_idx, signal.size))
    end_idx = max(0, min(end_idx, signal.size))

    if start_idx >= signal.size:
        return np.array([], dtype=float)

    if end_idx <= start_idx:
        end_idx = min(signal.size, start_idx + 1)

    return signal[start_idx:end_idx]


def _time_mask(times: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Return a mask for the requested interval, ensuring at least one time bin."""
    if times.size == 0:
        return np.array([], dtype=bool)

    mask = (times >= start_s) & (times < end_s)
    if mask.any():
        return mask

    nearest_idx = int(np.argmin(np.abs(times - ((start_s + end_s) / 2))))
    mask = np.zeros(times.shape, dtype=bool)
    mask[nearest_idx] = True
    return mask


def _band_summary(values: np.ndarray) -> tuple[float | None, float | None]:
    """Return mean/std summaries for an array of interval values."""
    if values.size == 0:
        return None, None

    return _optional_float(np.nanmean(values)), _optional_float(np.nanstd(values))


def _count_bursts(signal: np.ndarray, threshold: float, min_run_length: int) -> int:
    """Count contiguous high-activity runs above a threshold."""
    if signal.size == 0:
        return 0

    above_threshold = signal >= threshold
    burst_count = 0
    run_length = 0

    for is_active in above_threshold:
        if is_active:
            run_length += 1
            continue

        if run_length >= min_run_length:
            burst_count += 1
        run_length = 0

    if run_length >= min_run_length:
        burst_count += 1

    return burst_count


def _score_counts(scores: np.ndarray) -> tuple[dict[str, int], str | None]:
    """Summarize per-second scores into label counts and a dominant stage."""
    counts = {
        "Wake": 0,
        "NREM": 0,
        "REM": 0,
        "Unscored": 0,
    }

    for score in scores:
        if np.isnan(score) or score < 0:
            counts["Unscored"] += 1
            continue

        label = SLEEP_STAGE_LOOKUP.get(int(round(score)))
        if label is None:
            counts["Unscored"] += 1
            continue
        counts[label] += 1

    dominant_state = None
    scored_counts = {label: count for label, count in counts.items() if label != "Unscored"}
    if any(scored_counts.values()):
        dominant_state = max(scored_counts, key=scored_counts.get)

    return counts, dominant_state


def _score_to_label(score: float) -> str:
    """Map a raw score value to a human-readable stage label."""
    if np.isnan(score) or score < 0:
        return "Unscored"

    return SLEEP_STAGE_LOOKUP.get(int(round(score)), "Unscored")


def _spectral_interval_data(
    mat: dict[str, Any],
    start_s: float,
    end_s: float,
    fig: FigureResampler | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return spectrogram bins, theta/delta bins, and the source used."""
    spectrogram_trace = _find_trace(fig, trace_name="Spectrogram", fallback_index=-5)
    theta_delta_trace = _find_trace(fig, trace_name="Theta/Delta", fallback_index=-4)
    spectral_source = "figure"

    if spectrogram_trace is None or theta_delta_trace is None:
        eeg = _as_1d_float_array(mat.get("eeg"))
        eeg_frequency = _scalar_value(mat.get("eeg_frequency"))
        recording_start = _scalar_value(mat.get("start_time"), default=0.0)
        spectrogram_trace, theta_delta_trace = get_fft_plots(eeg, eeg_frequency, recording_start)
        spectral_source = "recomputed"

    spectrogram_times = _as_1d_float_array(getattr(spectrogram_trace, "x", None))
    spectrogram_freqs = _as_1d_float_array(getattr(spectrogram_trace, "y", None))
    spectrogram_values = np.asarray(getattr(spectrogram_trace, "z", []), dtype=float)
    theta_delta_times = _as_1d_float_array(getattr(theta_delta_trace, "x", None))
    theta_delta_values = _as_1d_float_array(getattr(theta_delta_trace, "y", None))

    spectrogram_mask = _time_mask(spectrogram_times, start_s, end_s)
    theta_delta_mask = _time_mask(theta_delta_times, start_s, end_s)

    if spectrogram_values.ndim != 2:
        spectrogram_values = np.empty((0, 0), dtype=float)

    interval_spectrogram = spectrogram_values[:, spectrogram_mask]
    interval_theta_delta = theta_delta_values[theta_delta_mask]

    return spectrogram_freqs, interval_spectrogram, interval_theta_delta, spectral_source


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
    fig: FigureResampler | None = None,
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
    interval_start, interval_end = _clamp_interval(mat, start_s, end_s)
    duration_s = interval_end - interval_start
    recording_start, _ = _get_recording_bounds(mat)

    spectrogram_freqs, interval_spectrogram, interval_theta_delta, spectral_source = (
        _spectral_interval_data(
            mat,
            interval_start,
            interval_end,
            fig,
        )
    )
    delta_mask = (spectrogram_freqs > 1) & (spectrogram_freqs <= 4)
    theta_mask = (spectrogram_freqs > 4) & (spectrogram_freqs <= 8)

    delta_power_mean_db, delta_power_std_db = _band_summary(interval_spectrogram[delta_mask])
    theta_power_mean_db, theta_power_std_db = _band_summary(interval_spectrogram[theta_mask])
    theta_delta_ratio_mean_db, theta_delta_ratio_std_db = _band_summary(interval_theta_delta)

    emg = _as_1d_float_array(mat.get("emg"))
    emg_frequency = _scalar_value(mat.get("eeg_frequency"))
    emg_segment = _slice_signal(
        emg,
        emg_frequency,
        interval_start,
        interval_end,
        recording_start,
    )
    emg_abs_segment = np.abs(emg_segment)
    emg_abs_full = np.abs(emg)
    emg_burst_threshold = _optional_float(np.nanpercentile(emg_abs_full, 90)) or 0.0
    emg_burst_count = _count_bursts(
        emg_abs_segment,
        threshold=emg_burst_threshold,
        min_run_length=max(1, int(round(emg_frequency * 0.1))),
    )

    scores = get_padded_sleep_scores(mat).astype(float)
    score_start_idx = int(np.floor(interval_start - recording_start))
    score_end_idx = int(np.ceil(interval_end - recording_start))
    score_start_idx = max(0, min(score_start_idx, scores.size))
    score_end_idx = max(score_start_idx, min(score_end_idx, scores.size))
    interval_scores = scores[score_start_idx:score_end_idx]
    current_score_counts, dominant_state = _score_counts(interval_scores)

    ne = _as_1d_float_array(mat.get("ne"))
    ne_frequency = _scalar_value(mat.get("ne_frequency"))
    ne_segment = _slice_signal(
        ne,
        ne_frequency,
        interval_start,
        interval_end,
        recording_start,
    )

    ne_level_mean = None
    ne_level_std = None
    ne_slope_per_second = None
    ne_drop_score = None
    if ne_segment.size:
        ne_level_mean = _optional_float(np.nanmean(ne_segment))
        ne_level_std = _optional_float(np.nanstd(ne_segment))
        if ne_segment.size >= 2:
            ne_slope_per_second = _optional_float(
                (ne_segment[-1] - ne_segment[0]) / max(duration_s, np.finfo(float).eps)
            )
            edge_window = max(1, ne_segment.size // 4)
            ne_drop_score = _optional_float(
                np.nanmean(ne_segment[:edge_window]) - np.nanmean(ne_segment[-edge_window:])
            )

    return {
        "start_s": int(math.floor(interval_start)),
        "end_s": int(math.ceil(interval_end)),
        "duration_s": _optional_float(duration_s),
        "spectral_source": spectral_source,
        "spectrogram_frequency_bin_count": int(spectrogram_freqs.size),
        "spectrogram_time_bin_count": (
            int(interval_spectrogram.shape[1]) if interval_spectrogram.ndim == 2 else 0
        ),
        "delta_power_mean_db": delta_power_mean_db,
        "delta_power_std_db": delta_power_std_db,
        "theta_power_mean_db": theta_power_mean_db,
        "theta_power_std_db": theta_power_std_db,
        "theta_delta_ratio_mean_db": theta_delta_ratio_mean_db,
        "theta_delta_ratio_std_db": theta_delta_ratio_std_db,
        "emg_rms": (
            _optional_float(np.sqrt(np.nanmean(np.square(emg_segment))))
            if emg_segment.size
            else None
        ),
        "emg_abs_mean": (
            _optional_float(np.nanmean(emg_abs_segment)) if emg_abs_segment.size else None
        ),
        "emg_abs_p90": (
            _optional_float(np.nanpercentile(emg_abs_segment, 90)) if emg_abs_segment.size else None
        ),
        "emg_burst_count": int(emg_burst_count),
        "current_score_counts": current_score_counts,
        "current_score_dominant_state": dominant_state,
        "ne_level_mean": ne_level_mean,
        "ne_level_std": ne_level_std,
        "ne_slope_per_second": ne_slope_per_second,
        "ne_drop_score": ne_drop_score,
    }


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
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s.")

    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    if scores_array.size == 0:
        raise ValueError("scores must contain at least one value.")

    interval_start_idx = max(0, int(math.floor(start_s)))
    interval_end_idx = min(scores_array.size, int(math.ceil(end_s)))

    if interval_end_idx <= interval_start_idx:
        raise ValueError("Requested interval falls outside the available scores.")

    interval_scores = scores_array[interval_start_idx:interval_end_idx]
    current_score_counts, dominant_state = _score_counts(interval_scores)

    raw_scores: list[dict[str, Any]] = []
    score_blocks: list[dict[str, Any]] = []

    for offset, score in enumerate(interval_scores):
        second = interval_start_idx + offset
        state = _score_to_label(score)
        score_value = None if state == "Unscored" else int(round(float(score)))
        raw_scores.append(
            {
                "second": second,
                "state": state,
                "score": score_value,
            }
        )

        if score_blocks and score_blocks[-1]["state"] == state:
            score_blocks[-1]["end_s"] = second + 1
            score_blocks[-1]["duration_s"] += 1
            continue

        score_blocks.append(
            {
                "start_s": second,
                "end_s": second + 1,
                "duration_s": 1,
                "state": state,
                "score": score_value,
            }
        )

    return {
        "start_s": interval_start_idx,
        "end_s": interval_end_idx,
        "duration_s": interval_end_idx - interval_start_idx,
        "scores": raw_scores,
        "score_blocks": score_blocks,
        "current_score_counts": current_score_counts,
        "current_score_dominant_state": dominant_state,
    }


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
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s.")

    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    if scores_array.size == 0:
        raise ValueError("scores must contain at least one value.")

    if state not in SLEEP_STAGE_TO_SCORE:
        raise ValueError(f"Unsupported sleep state: {state!r}.")

    interval_start_idx = max(0, int(math.floor(start_s)))
    interval_end_idx = min(scores_array.size, int(math.ceil(end_s)))

    if interval_end_idx <= interval_start_idx:
        raise ValueError("Requested interval falls outside the available scores.")

    updated_scores = scores_array.copy()
    updated_scores[interval_start_idx:interval_end_idx] = SLEEP_STAGE_TO_SCORE[state]
    return updated_scores


def apply_transition_rules(scores: np.ndarray) -> np.ndarray:
    """
    Return scores unchanged after validating the array shape.

    For the ChatGPT path, transition rules should primarily live in the model
    instructions so the model can reason about global bout structure while it
    scores. We intentionally avoid local post-hoc rewrites here because
    mechanical fixes can cascade into unintended neighboring changes.
    """
    scores_array = np.asarray(scores, dtype=float).reshape(-1)
    if scores_array.size == 0:
        raise ValueError("scores must contain at least one value.")

    return scores_array.copy()


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
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s.")

    reason_text = str(reason).strip()
    if not reason_text:
        raise ValueError("reason must be a non-empty string.")

    interval_start = int(math.floor(start_s))
    interval_end = int(math.ceil(end_s))
    if interval_end <= interval_start:
        interval_end = interval_start + 1

    return {
        "start_s": interval_start,
        "end_s": interval_end,
        "duration_s": interval_end - interval_start,
        "reason": reason_text,
    }
