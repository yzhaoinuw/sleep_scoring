"""Visualize Wake-only bouts from a low-band spectrogram threshold.

The script uses the same spectrogram builder as the app, normalizes the
displayed heatmap values globally, averages the 1-5 Hz rows column-wise, and
marks columns as Wake candidates when the feature crosses a threshold.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_src.config import FIX_NE_Y_RANGE, SPECTROGRAM_COLORSCALE  # noqa: E402
from app_src.get_fft_plots import get_fft_plots  # noqa: E402
from app_src.make_figure_dev import (  # noqa: E402
    COLORSCALE,
    HEATMAP_WIDTH,
    RANGE_PADDING_PERCENT,
    RANGE_QUANTILE,
    SLEEP_SCORE_OPACITY,
    STAGE_COLORS,
)


SPECTROGRAM_Y_MAX_HZ = 15
SPECTROGRAM_Y_TICKVALS_HZ = list(range(0, SPECTROGRAM_Y_MAX_HZ + 1, 5))
NTICKS = 24
PLOTLY_CONFIG = {"scrollZoom": True}
REM_STAGE_VALUE = 2


def _as_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(np.asarray(value).item())


def normalize_spectrogram_values(
    z: np.ndarray,
    lower_percentile: float | None = 5.0,
    upper_percentile: float | None = 95.0,
) -> tuple[np.ndarray, float, float]:
    """Normalize spectrogram values after optional percentile clipping."""
    if lower_percentile is None or upper_percentile is None:
        z_min = np.nanmin(z)
        z_max = np.nanmax(z)
    else:
        if not 0 <= lower_percentile < upper_percentile <= 100:
            raise ValueError(
                "Normalization percentiles must satisfy "
                f"0 <= lower < upper <= 100, got {lower_percentile:g}, "
                f"{upper_percentile:g}."
            )
        z_min, z_max = np.nanpercentile(z, [lower_percentile, upper_percentile])

    z_range = z_max - z_min
    if not np.isfinite(z_range) or z_range == 0:
        return np.zeros_like(z, dtype=float), float(z_min), float(z_max)
    clipped_z = np.clip(z, z_min, z_max)
    return (clipped_z - z_min) / z_range, float(z_min), float(z_max)


def compute_spectrogram_feature(
    mat: dict,
    low_hz: float,
    high_hz: float,
    window_duration: float,
    normalization_lower_percentile: float | None = 5.0,
    normalization_upper_percentile: float | None = 95.0,
) -> tuple[
    go.Heatmap, go.Scatter, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]
]:
    """Return the app-style spectrogram and normalized low-band column means."""
    eeg = np.asarray(mat["eeg"]).flatten()
    eeg_frequency = _as_scalar(mat["eeg_frequency"])
    start_time = _as_scalar(mat.get("start_time"), default=0.0)

    spectrogram, theta_delta_ratio = get_fft_plots(
        eeg,
        eeg_frequency,
        start_time,
        window_duration=window_duration,
    )

    times = np.asarray(spectrogram.x, dtype=float)
    frequencies = np.asarray(spectrogram.y, dtype=float)
    z = np.asarray(spectrogram.z, dtype=float)
    normalized_z, normalization_min, normalization_max = normalize_spectrogram_values(
        z,
        lower_percentile=normalization_lower_percentile,
        upper_percentile=normalization_upper_percentile,
    )

    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        raise ValueError(
            f"No spectrogram frequency bins found in {low_hz:g}-{high_hz:g} Hz. "
            f"Available range is {frequencies.min():g}-{frequencies.max():g} Hz."
        )

    low_band_means = np.nanmean(normalized_z[band_mask, :], axis=0)
    return (
        spectrogram,
        theta_delta_ratio,
        times,
        frequencies[band_mask],
        low_band_means,
        frequencies,
        (normalization_min, normalization_max),
    )


def threshold_wake_columns(
    values: np.ndarray,
    threshold: float,
    wake_when: str,
) -> np.ndarray:
    if wake_when == "below":
        return values <= threshold
    if wake_when == "above":
        return values >= threshold
    raise ValueError(f"wake_when must be 'below' or 'above', got {wake_when!r}")


def spectrogram_column_edges(
    times: np.ndarray,
    start_time: float,
    end_time: float,
) -> np.ndarray:
    if times.size == 0:
        return np.array([], dtype=float)
    if times.size == 1:
        half_width = 0.5
        return np.array(
            [
                max(start_time, times[0] - half_width),
                min(end_time, times[0] + half_width),
            ],
            dtype=float,
        )

    midpoints = (times[:-1] + times[1:]) / 2
    first_edge = times[0] - (midpoints[0] - times[0])
    last_edge = times[-1] + (times[-1] - midpoints[-1])
    edges = np.concatenate([[first_edge], midpoints, [last_edge]])
    return np.clip(edges, start_time, end_time)


def wake_columns_to_bouts(
    wake_columns: np.ndarray,
    edges: np.ndarray,
) -> list[tuple[float, float]]:
    bouts = []
    run_start: int | None = None

    for index, is_wake in enumerate(wake_columns):
        if is_wake and run_start is None:
            run_start = index
        elif not is_wake and run_start is not None:
            bouts.append((float(edges[run_start]), float(edges[index])))
            run_start = None

    if run_start is not None:
        bouts.append((float(edges[run_start]), float(edges[wake_columns.size])))

    return bouts


def merge_relative_nrem_gaps_once(
    wake_columns: np.ndarray,
    edges: np.ndarray,
    nrem_gap_merge_ratio: float = 0.5,
    max_nrem_gap_s: float | None = None,
) -> np.ndarray:
    """Convert NREM gaps to Wake when they are small relative to neighboring Wake."""
    if not 0 <= nrem_gap_merge_ratio:
        raise ValueError(
            "NREM gap merge ratio must be non-negative, " f"got {nrem_gap_merge_ratio:g}."
        )
    if max_nrem_gap_s is not None and max_nrem_gap_s < 0:
        raise ValueError(f"Max NREM gap duration must be non-negative, got {max_nrem_gap_s:g}.")

    wake_columns = np.asarray(wake_columns, dtype=bool).copy()
    if wake_columns.size == 0 or nrem_gap_merge_ratio == 0:
        return wake_columns
    if edges.size != wake_columns.size + 1:
        raise ValueError(
            "Edges must contain one more value than wake_columns, "
            f"got {edges.size} edges for {wake_columns.size} columns."
        )

    change_points = np.flatnonzero(np.diff(wake_columns)) + 1
    starts = np.r_[0, change_points]
    ends = np.r_[change_points, wake_columns.size]
    is_wake_run = wake_columns[starts]
    durations = edges[ends] - edges[starts]

    previous_is_wake = np.r_[False, is_wake_run[:-1]]
    next_is_wake = np.r_[is_wake_run[1:], False]
    previous_duration = np.r_[0.0, durations[:-1]]
    next_duration = np.r_[durations[1:], 0.0]

    neighboring_wake_duration = np.zeros_like(durations, dtype=float)
    neighboring_wake_duration += np.where(previous_is_wake, previous_duration, 0.0)
    neighboring_wake_duration += np.where(next_is_wake, next_duration, 0.0)

    merge_runs = ~is_wake_run
    merge_runs &= neighboring_wake_duration > 0
    merge_runs &= durations < nrem_gap_merge_ratio * neighboring_wake_duration
    if max_nrem_gap_s is not None:
        merge_runs &= durations <= max_nrem_gap_s

    for start, end in zip(starts[merge_runs], ends[merge_runs]):
        wake_columns[start:end] = True

    return wake_columns


def postprocess_wake_bouts(
    bouts: list[tuple[float, float]],
    min_bout_duration_s: float = 5.0,
) -> list[tuple[float, float]]:
    """Remove isolated short Wake bouts."""
    if min_bout_duration_s > 0:
        bouts = [(start, end) for start, end in bouts if end - start >= min_bout_duration_s]
    return bouts


def eeg_time_range(mat: dict) -> tuple[float, float]:
    eeg = np.asarray(mat["eeg"]).flatten()
    eeg_frequency = _as_scalar(mat["eeg_frequency"])
    start_time = _as_scalar(mat.get("start_time"), default=0.0)
    duration = math.ceil((eeg.size - 1) / eeg_frequency)
    return start_time, start_time + duration


def wake_bouts_to_sleep_score_heatmap(
    wake_bouts: list[tuple[float, float]],
    rem_bouts: list[tuple[float, float]] | None,
    start_time: float,
    end_time: float,
) -> list[list[float]]:
    """Create a one-row app-style sleep-score heatmap with Wake and optional REM."""
    duration = math.ceil(end_time - start_time)
    sleep_scores = np.full(duration, np.nan)
    for bout_start, bout_end in wake_bouts:
        start_index = max(0, math.floor(bout_start - start_time))
        end_index = min(duration, math.ceil(bout_end - start_time))
        sleep_scores[start_index:end_index] = 0
    for bout_start, bout_end in rem_bouts or []:
        start_index = max(0, math.floor(bout_start - start_time))
        end_index = min(duration, math.ceil(bout_end - start_time))
        sleep_scores[start_index:end_index] = REM_STAGE_VALUE
    return np.expand_dims(sleep_scores, axis=0).tolist()


def make_wake_sleep_score_trace(
    wake_bouts: list[tuple[float, float]],
    rem_bouts: list[tuple[float, float]] | None,
    start_time: float,
    end_time: float,
    num_class: int,
) -> go.Heatmap:
    """Build the Wake/REM overlay using the app's sleep-stage colors."""
    return go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,
        z=wake_bouts_to_sleep_score_heatmap(wake_bouts, rem_bouts, start_time, end_time),
        name="Threshold Wake/REM",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,
    )


def ne_time_axis(ne: np.ndarray, ne_frequency: float, start_time: float) -> np.ndarray:
    ne_end_time = (ne.size - 1) / ne_frequency + start_time
    return np.linspace(start_time, ne_end_time, ne.size)


def moving_average_ne(ne: np.ndarray, ne_frequency: float, window_s: float) -> np.ndarray:
    """Return a centered moving-average NE trace, preserving NaN gaps."""
    if window_s <= 0:
        return ne

    window_samples = max(1, int(round(window_s * ne_frequency)))
    if window_samples <= 1:
        return ne

    kernel = np.ones(window_samples, dtype=float)
    finite_mask = np.isfinite(ne)
    filled_ne = np.where(finite_mask, ne, 0.0)
    summed_ne = np.convolve(filled_ne, kernel, mode="same")
    sample_counts = np.convolve(finite_mask.astype(float), kernel, mode="same")

    smoothed_ne = np.full(ne.shape, np.nan, dtype=float)
    valid = sample_counts > 0
    smoothed_ne[valid] = summed_ne[valid] / sample_counts[valid]
    return smoothed_ne


def split_ne_segment_into_thirds(
    ne_segment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, middle, right = np.array_split(ne_segment, 3)
    return left, middle, right


def is_convex_ne_valley(
    ne_segment: np.ndarray, margin: float = 0.0
) -> tuple[bool, float, float, float]:
    """Return whether the middle third mean is lower than both outer third means."""
    if ne_segment.size < 3 or np.all(~np.isfinite(ne_segment)):
        return False, np.nan, np.nan, np.nan

    left, middle, right = split_ne_segment_into_thirds(ne_segment)
    left_mean = float(np.nanmean(left))
    middle_mean = float(np.nanmean(middle))
    right_mean = float(np.nanmean(right))
    is_convex = (
        np.isfinite(left_mean)
        and np.isfinite(middle_mean)
        and np.isfinite(right_mean)
        and middle_mean <= left_mean - margin
        and middle_mean <= right_mean - margin
    )
    return is_convex, left_mean, middle_mean, right_mean


def is_chord_ne_valley(ne_segment: np.ndarray, margin: float = 0.0) -> tuple[bool, float]:
    """Return whether every interior NE point is below the start-end chord."""
    if ne_segment.size < 3 or np.all(~np.isfinite(ne_segment)):
        return False, np.nan

    first_finite = 0
    while first_finite < ne_segment.size and not np.isfinite(ne_segment[first_finite]):
        first_finite += 1
    last_finite = ne_segment.size - 1
    while last_finite >= 0 and not np.isfinite(ne_segment[last_finite]):
        last_finite -= 1
    if last_finite - first_finite < 2:
        return False, np.nan

    segment = ne_segment[first_finite : last_finite + 1]
    chord = np.linspace(segment[0], segment[-1], segment.size)
    interior = segment[1:-1]
    interior_chord = chord[1:-1]
    finite_interior = np.isfinite(interior)
    if not np.any(finite_interior):
        return False, np.nan

    chord_minus_ne = interior_chord[finite_interior] - interior[finite_interior]
    is_chord_valley = bool(np.all(chord_minus_ne >= margin))
    min_chord_margin = float(np.nanmin(chord_minus_ne))
    return is_chord_valley, min_chord_margin


def classify_rem_bouts_from_wake_bouts(
    wake_bouts: list[tuple[float, float]],
    ne: np.ndarray | None,
    time_ne: np.ndarray | None,
    min_duration_s: float = 30.0,
    global_low_percentile: float = 10.0,
    low_ne_percentile: float = 0.0,
    shape_test: str = "thirds",
    convexity_margin: float = 0.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[dict[str, float | bool]]]:
    """Relabel Wake bouts as REM when NE is globally low and valley-shaped."""
    if ne is None or time_ne is None or ne.size == 0 or time_ne.size != ne.size:
        return wake_bouts, [], []
    if not 0 <= global_low_percentile <= 100:
        raise ValueError(
            "REM global low percentile must be between 0 and 100, "
            f"got {global_low_percentile:g}."
        )
    if not 0 <= low_ne_percentile <= 100:
        raise ValueError(
            "REM low NE bout percentile must be between 0 and 100, " f"got {low_ne_percentile:g}."
        )
    if shape_test not in {"chord", "thirds", "none"}:
        raise ValueError(
            f"REM shape test must be 'chord', 'thirds', or 'none', got {shape_test!r}."
        )

    finite_ne = ne[np.isfinite(ne)]
    if finite_ne.size == 0:
        return wake_bouts, [], []

    global_low_threshold = float(np.nanpercentile(finite_ne, global_low_percentile))
    remaining_wake_bouts = []
    rem_bouts = []
    diagnostics: list[dict[str, float | bool]] = []

    for bout_start, bout_end in wake_bouts:
        duration_s = bout_end - bout_start
        in_bout = (time_ne >= bout_start) & (time_ne < bout_end)
        ne_segment = ne[in_bout]
        has_segment = ne_segment.size > 0 and np.any(np.isfinite(ne_segment))
        min_ne = float(np.nanmin(ne_segment)) if has_segment else np.nan
        median_ne = float(np.nanmedian(ne_segment)) if has_segment else np.nan
        low_ne_value = (
            float(np.nanpercentile(ne_segment, low_ne_percentile)) if has_segment else np.nan
        )
        reaches_global_low = bool(has_segment and low_ne_value <= global_low_threshold)
        thirds_is_convex, left_mean, middle_mean, right_mean = is_convex_ne_valley(
            ne_segment,
            margin=convexity_margin,
        )
        chord_is_convex, chord_min_margin = is_chord_ne_valley(
            ne_segment,
            margin=convexity_margin,
        )
        if shape_test == "chord":
            is_convex = chord_is_convex
        elif shape_test == "thirds":
            is_convex = thirds_is_convex
        else:
            is_convex = True
        is_rem = bool(duration_s >= min_duration_s and reaches_global_low and is_convex)

        diagnostics.append(
            {
                "start": float(bout_start),
                "end": float(bout_end),
                "duration_s": float(duration_s),
                "min_ne": min_ne,
                "median_ne": median_ne,
                "low_ne_percentile": float(low_ne_percentile),
                "low_ne_value": low_ne_value,
                "global_low_threshold": global_low_threshold,
                "left_mean": left_mean,
                "middle_mean": middle_mean,
                "right_mean": right_mean,
                "chord_min_margin": chord_min_margin,
                "shape_test": shape_test,
                "reaches_global_low": reaches_global_low,
                "thirds_is_convex": bool(thirds_is_convex),
                "chord_is_convex": bool(chord_is_convex),
                "is_convex": bool(is_convex),
                "is_rem": is_rem,
            }
        )

        if is_rem:
            rem_bouts.append((bout_start, bout_end))
        else:
            remaining_wake_bouts.append((bout_start, bout_end))

    return remaining_wake_bouts, rem_bouts, diagnostics


def split_rem_bouts_at_ne_recovery(
    rem_bouts: list[tuple[float, float]],
    ne: np.ndarray | None,
    time_ne: np.ndarray | None,
    epsilon_fraction: float = 0.02,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[dict[str, float | bool]],
]:
    """Split post-trough NE recovery tails from REM bouts into Wake."""
    if epsilon_fraction < 0:
        raise ValueError(
            f"REM recovery epsilon fraction must be non-negative, got {epsilon_fraction:g}."
        )
    if ne is None or time_ne is None or ne.size == 0 or time_ne.size != ne.size:
        return rem_bouts, [], []

    split_rem_bouts = []
    recovery_wake_bouts = []
    diagnostics: list[dict[str, float | bool]] = []

    for bout_start, bout_end in rem_bouts:
        in_bout = (time_ne >= bout_start) & (time_ne < bout_end)
        segment = ne[in_bout]
        segment_time = time_ne[in_bout]
        finite_mask = np.isfinite(segment)
        has_segment = segment.size > 1 and np.any(finite_mask)
        split_found = False
        split_time = np.nan
        trough_time = np.nan
        epsilon = np.nan

        if has_segment:
            finite_indices = np.flatnonzero(finite_mask)
            trough_local_index = finite_indices[int(np.nanargmin(segment[finite_mask]))]
            trough_time = float(segment_time[trough_local_index])
            post_trough = segment[trough_local_index:]
            finite_post = post_trough[np.isfinite(post_trough)]
            if finite_post.size > 1:
                ne_range = float(np.nanpercentile(finite_post, 90) - np.nanmin(finite_post))
                epsilon = epsilon_fraction * ne_range if np.isfinite(ne_range) else 0.0
                cumulative_diff = np.r_[0.0, np.cumsum(np.diff(segment))]
                recovery_indices = np.flatnonzero(cumulative_diff[trough_local_index:] > epsilon)
                if recovery_indices.size > 0:
                    split_index = int(trough_local_index + recovery_indices[0])
                    split_time = float(segment_time[split_index])
                    split_found = bool(bout_start < split_time < bout_end)

        diagnostics.append(
            {
                "start": float(bout_start),
                "end": float(bout_end),
                "trough_time": trough_time,
                "epsilon": epsilon,
                "split_time": split_time,
                "split_found": split_found,
            }
        )

        if split_found:
            split_rem_bouts.append((bout_start, split_time))
            recovery_wake_bouts.append((split_time, bout_end))
        else:
            split_rem_bouts.append((bout_start, bout_end))

    return split_rem_bouts, recovery_wake_bouts, diagnostics


def make_wake_bout_figure(
    mat: dict,
    mat_name: str,
    threshold: float,
    wake_when: str = "below",
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
    normalization_lower_percentile: float | None = 5.0,
    normalization_upper_percentile: float | None = 95.0,
    postprocess_wake_bouts_enabled: bool = True,
    min_bout_duration_s: float = 5.0,
    nrem_gap_merge_ratio: float = 0.5,
    max_nrem_gap_s: float | None = None,
    detect_rem_bouts: bool = True,
    rem_min_bout_duration_s: float = 30.0,
    rem_global_low_percentile: float = 10.0,
    rem_low_ne_percentile: float = 0.0,
    rem_smoothing_window_s: float = 5.0,
    rem_shape_test: str = "thirds",
    rem_convexity_margin: float = 0.0,
    rem_recovery_epsilon_fraction: float = 0.02,
    default_n_shown_samples: int = 2048,
    num_class: int = 3,
) -> tuple[go.Figure, list[tuple[float, float]], list[tuple[float, float]], np.ndarray]:
    start_time, end_time = eeg_time_range(mat)
    if mat.get("num_class") is not None:
        num_class = int(_as_scalar(mat["num_class"]))
    if detect_rem_bouts:
        num_class = max(num_class, 3)
    if num_class not in COLORSCALE:
        num_class = 3

    (
        spectrogram,
        theta_delta_ratio,
        times,
        band_frequencies,
        low_band_means,
        _frequencies,
        normalization_range,
    ) = compute_spectrogram_feature(
        mat,
        low_hz=low_hz,
        high_hz=high_hz,
        window_duration=window_duration,
        normalization_lower_percentile=normalization_lower_percentile,
        normalization_upper_percentile=normalization_upper_percentile,
    )
    wake_columns = threshold_wake_columns(
        low_band_means,
        threshold=threshold,
        wake_when=wake_when,
    )
    edges = spectrogram_column_edges(times, start_time=start_time, end_time=end_time)
    raw_wake_columns = wake_columns
    if postprocess_wake_bouts_enabled:
        wake_columns = merge_relative_nrem_gaps_once(
            wake_columns,
            edges,
            nrem_gap_merge_ratio=nrem_gap_merge_ratio,
            max_nrem_gap_s=max_nrem_gap_s,
        )
    raw_wake_bouts = wake_columns_to_bouts(raw_wake_columns, edges)
    merged_wake_bouts = wake_columns_to_bouts(wake_columns, edges)
    wake_bouts = postprocess_wake_bouts(
        merged_wake_bouts,
        min_bout_duration_s=min_bout_duration_s,
    )
    wake_bouts_before_rem = list(wake_bouts)

    fig = FigureResampler(
        make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.055,
            subplot_titles=("EEG Spectrogram", "NE", "Smoothed NE for REM"),
            row_heights=[0.40, 0.30, 0.30],
            specs=[
                [{"secondary_y": True, "r": -0.05}],
                [{"r": -0.05}],
                [{"r": -0.05}],
            ],
        ),
        default_n_shown_samples=default_n_shown_samples,
        default_downsampler=MinMaxLTTB(parallel=True),
    )

    spectrogram.colorscale = SPECTROGRAM_COLORSCALE
    spectrogram.colorbar = dict(
        title="Power (dB)",
        orientation="h",
        thicknessmode="fraction",
        thickness=0.02,
        lenmode="fraction",
        len=0.18,
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=0.8,
        tickfont=dict(size=8),
    )
    ne = mat.get("ne")
    ne_frequency = mat.get("ne_frequency")
    ne_range = 1.0
    time_ne = None
    ne_for_rem = None
    if ne is not None and ne_frequency is not None and np.asarray(ne).size > 1:
        ne = np.asarray(ne).flatten()
        ne_frequency = _as_scalar(ne_frequency)
        time_ne = ne_time_axis(ne, ne_frequency, start_time)
        ne_for_rem = moving_average_ne(ne, ne_frequency, rem_smoothing_window_s)
        ne_lower_range, ne_upper_range = (
            np.nanquantile(ne, 1 - RANGE_QUANTILE),
            np.nanquantile(ne, RANGE_QUANTILE),
        )
        ne_range = max(abs(ne_lower_range), abs(ne_upper_range))
        if not np.isfinite(ne_range) or ne_range == 0:
            ne_range = 1.0
        fig.add_trace(
            go.Scattergl(
                name="NE",
                line=dict(width=1),
                marker=dict(size=2, color="black"),
                showlegend=False,
                mode="lines+markers",
                hovertemplate="<b>time</b>: %{x:.2f}<br><b>y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=ne,
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                name=f"Smoothed NE ({rem_smoothing_window_s:g}s)",
                line=dict(width=1.5, color="rgb(31, 119, 180)"),
                marker=dict(size=2, color="rgb(31, 119, 180)"),
                showlegend=False,
                mode="lines",
                hovertemplate="<b>time</b>: %{x:.2f}<br><b>smoothed y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=ne_for_rem,
            row=3,
            col=1,
        )
    else:
        fig.add_annotation(
            text="NE unavailable",
            x=0.5,
            xref="paper",
            y=0.5,
            yref="y2 domain",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )
        fig.add_annotation(
            text="Smoothed NE unavailable",
            x=0.5,
            xref="paper",
            y=0.5,
            yref="y3 domain",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )
        ne = None

    rem_bouts: list[tuple[float, float]] = []
    rem_diagnostics: list[dict[str, float | bool]] = []
    rem_recovery_diagnostics: list[dict[str, float | bool]] = []
    if detect_rem_bouts:
        wake_bouts, rem_bouts, rem_diagnostics = classify_rem_bouts_from_wake_bouts(
            wake_bouts,
            ne=ne_for_rem,
            time_ne=time_ne,
            min_duration_s=rem_min_bout_duration_s,
            global_low_percentile=rem_global_low_percentile,
            low_ne_percentile=rem_low_ne_percentile,
            shape_test=rem_shape_test,
            convexity_margin=rem_convexity_margin,
        )
        rem_bouts, recovery_wake_bouts, rem_recovery_diagnostics = split_rem_bouts_at_ne_recovery(
            rem_bouts,
            ne=ne_for_rem,
            time_ne=time_ne,
            epsilon_fraction=rem_recovery_epsilon_fraction,
        )
        wake_bouts = sorted(wake_bouts + recovery_wake_bouts)

    wake_sleep_scores = make_wake_sleep_score_trace(
        wake_bouts,
        rem_bouts=rem_bouts,
        start_time=start_time,
        end_time=end_time,
        num_class=num_class,
    )

    fig.add_trace(spectrogram, secondary_y=False, row=1, col=1)
    fig.add_trace(theta_delta_ratio, secondary_y=True, row=1, col=1)
    fig.add_trace(wake_sleep_scores, row=2, col=1)
    fig.add_trace(wake_sleep_scores, row=3, col=1)

    total_wake_s = sum(end - start for start, end in wake_bouts)
    total_rem_s = sum(end - start for start, end in rem_bouts)
    max_gap_text = "no max gap" if max_nrem_gap_s is None else f"max NREM gap {max_nrem_gap_s:g}s"
    cleanup_text = (
        f"cleanup on: NREM < {nrem_gap_merge_ratio:g}x neighbor Wake sum, "
        f"{max_gap_text}, remove Wake < {min_bout_duration_s:g}s"
        if postprocess_wake_bouts_enabled
        else "cleanup off"
    )
    rem_text = (
        f"REM on: Wake >= {rem_min_bout_duration_s:g}s, "
        f"bout NE p{rem_low_ne_percentile:g} <= global p{rem_global_low_percentile:g}, "
        f"{rem_smoothing_window_s:g}s smooth, {rem_shape_test} shape, recovery eps "
        f"{rem_recovery_epsilon_fraction:g}x"
        if detect_rem_bouts
        else "REM off"
    )
    normalization_text = format_normalization_label(
        normalization_lower_percentile,
        normalization_upper_percentile,
    )
    fig.update_layout(
        autosize=True,
        margin=dict(t=70, l=10, r=5, b=25),
        height=820,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=(
                f"{mat_name}: Wake/REM low-band threshold view"
                f"<br><sup>{low_hz:g}-{high_hz:g} Hz mean {wake_when} "
                f"{threshold:g}; {normalization_text}; {cleanup_text}; "
                f"{rem_text}; Wake {len(wake_bouts)} bouts/{total_wake_s:.1f} s; "
                f"REM {len(rem_bouts)} bouts/{total_rem_s:.1f} s</sup>"
            ),
            font=dict(size=16),
            xanchor="left",
            x=0.03,
            automargin=True,
            yref="paper",
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
        xaxis3=dict(
            tickformat="digits",
            nticks=NTICKS,
            tickfont=dict(size=10),
            automargin=True,
        ),
    )
    fig.update_traces(xaxis="x3")
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(range=[start_time, end_time], row=2, col=1)
    fig.update_xaxes(
        range=[start_time, end_time],
        row=3,
        col=1,
        title_text="<b>Time (s)</b>",
        title_standoff=10,
        ticklabelstandoff=5,
    )
    fig.update_yaxes(
        title="Frequency (Hz)",
        range=[0, SPECTROGRAM_Y_MAX_HZ],
        tickmode="array",
        tickvals=SPECTROGRAM_Y_TICKVALS_HZ,
        fixedrange=True,
        secondary_y=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title="Theta/Delta",
        overlaying="y",
        side="right",
        fixedrange=True,
        secondary_y=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + RANGE_PADDING_PERCENT),
            ne_range * (1 + RANGE_PADDING_PERCENT),
        ],
        fixedrange=FIX_NE_Y_RANGE,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[
            ne_range * -(1 + RANGE_PADDING_PERCENT),
            ne_range * (1 + RANGE_PADDING_PERCENT),
        ],
        fixedrange=FIX_NE_Y_RANGE,
        row=3,
        col=1,
    )
    fig.update_annotations(font_size=14)
    if fig["layout"]["annotations"]:
        fig["layout"]["annotations"][-1]["font"]["size"] = 14

    print(
        f"Used {band_frequencies.size} frequency bins from "
        f"{band_frequencies.min():.3g}-{band_frequencies.max():.3g} Hz."
    )
    print(
        f"Normalization: {normalization_text}, "
        f"value range {normalization_range[0]:.4g} to {normalization_range[1]:.4g}"
    )
    print(f"Wake overlay color: {STAGE_COLORS[0]}")
    print(f"REM overlay color: {STAGE_COLORS[REM_STAGE_VALUE]}")
    print(
        f"Wake candidates: {len(raw_wake_bouts)} raw -> {len(wake_bouts_before_rem)} "
        f"({'postprocessed' if postprocess_wake_bouts_enabled else 'raw'})"
    )
    if postprocess_wake_bouts_enabled:
        print(
            f"Post-processing: {len(raw_wake_bouts)} raw Wake bout(s) -> "
            f"{len(merged_wake_bouts)} after relative NREM-gap merging -> "
            f"{len(wake_bouts_before_rem)} after short-Wake removal."
        )
    if detect_rem_bouts:
        evaluated_rem_candidates = len(rem_diagnostics)
        print(
            f"REM detection: on, {len(rem_bouts)} REM bout(s) from "
            f"{evaluated_rem_candidates} Wake candidate(s)."
        )
        if rem_diagnostics:
            threshold_value = rem_diagnostics[0]["global_low_threshold"]
            print(
                f"REM NE bout p{rem_low_ne_percentile:g} <= global p{rem_global_low_percentile:g} "
                f"threshold after {rem_smoothing_window_s:g}s smoothing: {threshold_value:.4g}"
            )
        split_count = sum(1 for item in rem_recovery_diagnostics if item["split_found"])
        print(
            f"REM recovery split: {split_count}/{len(rem_recovery_diagnostics)} REM bout(s), "
            f"epsilon {rem_recovery_epsilon_fraction:g}x post-trough p90 range."
        )
    else:
        print("REM detection: off")
    return fig, wake_bouts, rem_bouts, low_band_means


def format_threshold_for_filename(threshold: float) -> str:
    threshold_text = f"{threshold:g}".replace("-", "neg_").replace(".", "p")
    return threshold_text


def format_seconds_for_filename(seconds: float) -> str:
    return f"{seconds:g}".replace("-", "neg_").replace(".", "p")


def format_normalization_label(
    lower_percentile: float | None,
    upper_percentile: float | None,
) -> str:
    if lower_percentile is None or upper_percentile is None:
        return "normalization min-max"
    return f"normalization p{lower_percentile:g}-p{upper_percentile:g} clipped"


def format_normalization_for_filename(
    lower_percentile: float | None,
    upper_percentile: float | None,
) -> str:
    if lower_percentile is None or upper_percentile is None:
        return "norm_minmax"
    lower_text = format_seconds_for_filename(lower_percentile)
    upper_text = format_seconds_for_filename(upper_percentile)
    return f"norm_p{lower_text}_p{upper_text}"


def default_output_path(
    mat_path: Path,
    output_dir: Path | None,
    threshold: float,
    wake_when: str,
    normalization_lower_percentile: float | None,
    normalization_upper_percentile: float | None,
    postprocess_wake_bouts_enabled: bool,
    min_bout_duration_s: float,
    nrem_gap_merge_ratio: float,
    max_nrem_gap_s: float | None,
    detect_rem_bouts: bool,
    rem_min_bout_duration_s: float,
    rem_global_low_percentile: float,
    rem_low_ne_percentile: float,
    rem_smoothing_window_s: float,
    rem_shape_test: str,
    rem_convexity_margin: float,
    rem_recovery_epsilon_fraction: float,
) -> Path:
    base_dir = output_dir if output_dir is not None else mat_path.parent
    rem_text = (
        "rem"
        f"_bout_p{format_seconds_for_filename(rem_low_ne_percentile)}"
        f"_global_p{format_seconds_for_filename(rem_global_low_percentile)}"
        f"_smooth_{format_seconds_for_filename(rem_smoothing_window_s)}s"
        f"_{rem_shape_test}"
        f"_recover_eps_{format_seconds_for_filename(rem_recovery_epsilon_fraction)}x"
        if detect_rem_bouts
        else "wake_only"
    )
    return base_dir / f"{mat_path.stem}_low_band_wake_bouts_{rem_text}.html"


def run_wake_bout_visualization(
    mat_file: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    threshold: float = 0.8,
    wake_when: str = "below",
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
    normalization_lower_percentile: float | None = 5.0,
    normalization_upper_percentile: float | None = 95.0,
    postprocess_wake_bouts_enabled: bool = True,
    min_bout_duration_s: float = 5.0,
    nrem_gap_merge_ratio: float = 0.5,
    max_nrem_gap_s: float | None = None,
    detect_rem_bouts: bool = True,
    rem_min_bout_duration_s: float = 30.0,
    rem_global_low_percentile: float = 10.0,
    rem_low_ne_percentile: float = 0.0,
    rem_smoothing_window_s: float = 5.0,
    rem_shape_test: str = "thirds",
    rem_convexity_margin: float = 0.0,
    rem_recovery_epsilon_fraction: float = 0.02,
    default_n_shown_samples: int = 2048,
    show: bool = False,
) -> Path:
    mat_path = mat_file.expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(mat_path, squeeze_me=True)
    fig, wake_bouts, rem_bouts, low_band_means = make_wake_bout_figure(
        mat,
        mat_name=mat_path.name,
        threshold=threshold,
        wake_when=wake_when,
        low_hz=low_hz,
        high_hz=high_hz,
        window_duration=window_duration,
        normalization_lower_percentile=normalization_lower_percentile,
        normalization_upper_percentile=normalization_upper_percentile,
        postprocess_wake_bouts_enabled=postprocess_wake_bouts_enabled,
        min_bout_duration_s=min_bout_duration_s,
        nrem_gap_merge_ratio=nrem_gap_merge_ratio,
        max_nrem_gap_s=max_nrem_gap_s,
        detect_rem_bouts=detect_rem_bouts,
        rem_min_bout_duration_s=rem_min_bout_duration_s,
        rem_global_low_percentile=rem_global_low_percentile,
        rem_low_ne_percentile=rem_low_ne_percentile,
        rem_smoothing_window_s=rem_smoothing_window_s,
        rem_shape_test=rem_shape_test,
        rem_convexity_margin=rem_convexity_margin,
        rem_recovery_epsilon_fraction=rem_recovery_epsilon_fraction,
        default_n_shown_samples=default_n_shown_samples,
    )

    output_path = (
        output.expanduser().resolve()
        if output is not None
        else default_output_path(
            mat_path,
            output_dir,
            threshold=threshold,
            wake_when=wake_when,
            normalization_lower_percentile=normalization_lower_percentile,
            normalization_upper_percentile=normalization_upper_percentile,
            postprocess_wake_bouts_enabled=postprocess_wake_bouts_enabled,
            min_bout_duration_s=min_bout_duration_s,
            nrem_gap_merge_ratio=nrem_gap_merge_ratio,
            max_nrem_gap_s=max_nrem_gap_s,
            detect_rem_bouts=detect_rem_bouts,
            rem_min_bout_duration_s=rem_min_bout_duration_s,
            rem_global_low_percentile=rem_global_low_percentile,
            rem_low_ne_percentile=rem_low_ne_percentile,
            rem_smoothing_window_s=rem_smoothing_window_s,
            rem_shape_test=rem_shape_test,
            rem_convexity_margin=rem_convexity_margin,
            rem_recovery_epsilon_fraction=rem_recovery_epsilon_fraction,
        ).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, config=PLOTLY_CONFIG)

    quantiles = np.nanquantile(low_band_means, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    print(f"Wrote {output_path}")
    print(
        f"Wake rule: normalized {low_hz:g}-{high_hz:g} Hz column mean " f"{wake_when} {threshold:g}"
    )
    print(
        "Feature normalization: "
        + format_normalization_label(
            normalization_lower_percentile,
            normalization_upper_percentile,
        )
    )
    print(f"Predicted Wake bouts: {len(wake_bouts)}")
    print(f"Predicted REM bouts: {len(rem_bouts)}")
    max_nrem_gap_label = "none" if max_nrem_gap_s is None else f"{max_nrem_gap_s:g}s"
    print(
        "Post-processing: "
        + (
            f"on, merge NREM gaps < {nrem_gap_merge_ratio:g}x neighboring Wake sum, "
            f"max NREM gap {max_nrem_gap_label}, "
            f"remove Wake bouts < {min_bout_duration_s:g}s"
            if postprocess_wake_bouts_enabled
            else "off"
        )
    )
    print(
        "REM detection: "
        + (
            f"on, Wake duration >= {rem_min_bout_duration_s:g}s, "
            f"bout NE p{rem_low_ne_percentile:g} <= global p{rem_global_low_percentile:g}, "
            f"smoothing {rem_smoothing_window_s:g}s, "
            f"{rem_shape_test} shape margin {rem_convexity_margin:g}, "
            f"recovery epsilon {rem_recovery_epsilon_fraction:g}x post-trough p90 range"
            if detect_rem_bouts
            else "off"
        )
    )
    print(
        "Column mean quantiles "
        "(min, p10, p25, median, p75, p90, max): "
        + ", ".join(f"{value:.4f}" for value in quantiles)
    )
    if show:
        fig.show(config=PLOTLY_CONFIG)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Threshold normalized 1-5 Hz spectrogram column means into Wake-only "
            "bouts and visualize them over EEG spectrogram + NE."
        )
    )
    parser.add_argument("mat_file", type=Path, help="Path to the .mat file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="HTML file to write. Defaults beside the .mat file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the default HTML filename. Ignored when --output is set.",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument(
        "--wake-when",
        choices=["below", "above"],
        default="below",
        help="Whether values below or above the threshold should be labeled Wake.",
    )
    parser.add_argument("--low-hz", type=float, default=1.0)
    parser.add_argument("--high-hz", type=float, default=5.0)
    parser.add_argument("--window-duration", type=float, default=5.0)
    parser.add_argument(
        "--normalization-lower-percentile",
        type=float,
        default=5.0,
        help="Lower percentile cap before min-max normalization.",
    )
    parser.add_argument(
        "--normalization-upper-percentile",
        type=float,
        default=95.0,
        help="Upper percentile cap before min-max normalization.",
    )
    parser.add_argument(
        "--use-raw-minmax-normalization",
        action="store_true",
        help="Disable percentile caps and normalize by raw min/max values.",
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="Plot raw threshold bouts without NREM-gap merging or tiny-Wake removal.",
    )
    parser.add_argument("--min-bout-duration-s", type=float, default=5.0)
    parser.add_argument(
        "--nrem-gap-merge-ratio",
        type=float,
        default=0.5,
        help=(
            "Convert a non-Wake/NREM gap to Wake when its duration is less than "
            "this ratio times the sum of neighboring Wake durations."
        ),
    )
    parser.add_argument(
        "--max-nrem-gap-s",
        type=float,
        default=None,
        help="Optional maximum NREM gap duration eligible for Wake merging.",
    )
    parser.add_argument(
        "--no-rem-detection",
        action="store_true",
        help="Disable the optional NE-based Wake-to-REM relabeling pass.",
    )
    parser.add_argument(
        "--rem-min-bout-duration-s",
        type=float,
        default=30.0,
        help="Minimum Wake bout duration considered for REM relabeling.",
    )
    parser.add_argument(
        "--rem-global-low-percentile",
        type=float,
        default=10.0,
        help="Global NE percentile a candidate bout must reach to become REM.",
    )
    parser.add_argument(
        "--rem-low-ne-percentile",
        type=float,
        default=0.0,
        help=(
            "Within-bout NE percentile compared against the global REM percentile. "
            "Use 0 for min-like behavior, 50 for median-like behavior."
        ),
    )
    parser.add_argument(
        "--rem-smoothing-window-s",
        type=float,
        default=5.0,
        help="Centered global moving-average window applied to NE before REM detection.",
    )
    parser.add_argument(
        "--rem-shape-test",
        choices=["chord", "thirds", "none"],
        default="thirds",
        help="NE valley shape test for REM candidates. 'thirds' compares third means; use 'none' to skip this gate.",
    )
    parser.add_argument(
        "--rem-convexity-margin",
        type=float,
        default=0.0,
        help="Required NE drop for the selected REM shape test.",
    )
    parser.add_argument(
        "--rem-recovery-epsilon-fraction",
        type=float,
        default=0.02,
        help=(
            "Post-REM Wake split threshold as a fraction of the post-trough NE p90 range. "
            "Use 0 to split at the first positive cumulative NE diff."
        ),
    )
    parser.add_argument("--default-n-shown-samples", type=int, default=2048)
    parser.add_argument("--show", action="store_true", help="Open the figure after writing HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalization_lower_percentile = (
        None if args.use_raw_minmax_normalization else args.normalization_lower_percentile
    )
    normalization_upper_percentile = (
        None if args.use_raw_minmax_normalization else args.normalization_upper_percentile
    )
    run_wake_bout_visualization(
        mat_file=args.mat_file,
        output=args.output,
        output_dir=args.output_dir,
        threshold=args.threshold,
        wake_when=args.wake_when,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        window_duration=args.window_duration,
        normalization_lower_percentile=normalization_lower_percentile,
        normalization_upper_percentile=normalization_upper_percentile,
        postprocess_wake_bouts_enabled=not args.no_postprocess,
        min_bout_duration_s=args.min_bout_duration_s,
        nrem_gap_merge_ratio=args.nrem_gap_merge_ratio,
        max_nrem_gap_s=args.max_nrem_gap_s,
        detect_rem_bouts=not args.no_rem_detection,
        rem_min_bout_duration_s=args.rem_min_bout_duration_s,
        rem_global_low_percentile=args.rem_global_low_percentile,
        rem_low_ne_percentile=args.rem_low_ne_percentile,
        rem_smoothing_window_s=args.rem_smoothing_window_s,
        rem_shape_test=args.rem_shape_test,
        rem_convexity_margin=args.rem_convexity_margin,
        rem_recovery_epsilon_fraction=args.rem_recovery_epsilon_fraction,
        default_n_shown_samples=args.default_n_shown_samples,
        show=args.show,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Editable parameters for direct console/Spyder-style runs.
        MAT_FILE = REPO_ROOT / "user_test_files" / "830.mat"
        OUTPUT = None  # Path to a specific .html file, or None for the default name.
        OUTPUT_DIR = REPO_ROOT / "test-artifacts"
        THRESHOLD = 0.7
        WAKE_WHEN = "below"  # "below" means low 1-5 Hz power marks Wake.
        LOW_HZ = 1.0
        HIGH_HZ = 7.0
        WINDOW_DURATION = 5.0
        NORMALIZATION_LOWER_PERCENTILE = 5.0
        NORMALIZATION_UPPER_PERCENTILE = 95.0
        POSTPROCESS_WAKE_BOUTS = True
        MIN_BOUT_DURATION_S = 5.0
        NREM_GAP_MERGE_RATIO = 0.5
        MAX_NREM_GAP_S = None
        DETECT_REM_BOUTS = True
        REM_MIN_BOUT_DURATION_S = 30.0
        REM_GLOBAL_LOW_PERCENTILE = 10.0
        REM_LOW_NE_PERCENTILE = 5.0
        REM_SMOOTHING_WINDOW_S = 10.0
        REM_SHAPE_TEST = "thirds"
        REM_CONVEXITY_MARGIN = 0.0
        REM_RECOVERY_EPSILON_FRACTION = 0.02
        DEFAULT_N_SHOWN_SAMPLES = 2048
        SHOW = False

        run_wake_bout_visualization(
            mat_file=MAT_FILE,
            output=OUTPUT,
            output_dir=OUTPUT_DIR,
            threshold=THRESHOLD,
            wake_when=WAKE_WHEN,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
            window_duration=WINDOW_DURATION,
            normalization_lower_percentile=NORMALIZATION_LOWER_PERCENTILE,
            normalization_upper_percentile=NORMALIZATION_UPPER_PERCENTILE,
            postprocess_wake_bouts_enabled=POSTPROCESS_WAKE_BOUTS,
            min_bout_duration_s=MIN_BOUT_DURATION_S,
            nrem_gap_merge_ratio=NREM_GAP_MERGE_RATIO,
            max_nrem_gap_s=MAX_NREM_GAP_S,
            detect_rem_bouts=DETECT_REM_BOUTS,
            rem_min_bout_duration_s=REM_MIN_BOUT_DURATION_S,
            rem_global_low_percentile=REM_GLOBAL_LOW_PERCENTILE,
            rem_low_ne_percentile=REM_LOW_NE_PERCENTILE,
            rem_smoothing_window_s=REM_SMOOTHING_WINDOW_S,
            rem_shape_test=REM_SHAPE_TEST,
            rem_convexity_margin=REM_CONVEXITY_MARGIN,
            rem_recovery_epsilon_fraction=REM_RECOVERY_EPSILON_FRACTION,
            default_n_shown_samples=DEFAULT_N_SHOWN_SAMPLES,
            show=SHOW,
        )
