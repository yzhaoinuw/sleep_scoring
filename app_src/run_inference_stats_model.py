"""Run the low-band statistical Wake/REM model.

This module is a trimmed copy of ``visualize_low_band_wake_bouts.py`` shaped
toward the app inference API. It keeps the current working statistical
pipeline, plus a developer diagnostic figure for direct console runs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly_resampler import FigureResampler
from plotly_resampler.aggregation import MinMaxLTTB


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_src.config import (  # noqa: E402
    FIX_NE_Y_RANGE,
    SPECTROGRAM_COLORSCALE,
    STATS_MODEL_MIN_REM_DURATION,
    STATS_MODEL_MIN_WAKE_DURATION,
    STATS_MODEL_WAKE_THRESHOLD,
)
from app_src.get_fft_plots import get_fft_plots  # noqa: E402
from app_src.make_figure_dev import (  # noqa: E402
    COLORSCALE,
    HEATMAP_WIDTH,
    RANGE_PADDING_PERCENT,
    RANGE_QUANTILE,
    SLEEP_SCORE_OPACITY,
    STAGE_COLORS,
    STAGE_NAMES,
)


@dataclass(frozen=True)
class StatsModelConfig:
    """Configuration for the statistical Wake/REM model.

    The first three user-facing controls are exposed in ``app_src.config``:
    ``wake_threshold``, ``min_wake_duration``, and ``min_rem_duration``.
    The remaining values stay internal for now.
    """

    wake_threshold: float = STATS_MODEL_WAKE_THRESHOLD
    spectrogram_sleep_wave_range: tuple[float, float] = (1.0, 7.0)
    spectrogram_normalization_range: tuple[float, float] = (5.0, 95.0)
    min_wake_duration: float = STATS_MODEL_MIN_WAKE_DURATION
    wake_merge_coefficient: float = 0.5
    min_rem_duration: float = STATS_MODEL_MIN_REM_DURATION
    rem_threshold_percentile: float = 10.0
    rem_threshold_comparison_percentile: float = 5.0
    ne_smoothing_window: float = 10.0
    rem_recovery_epsilon_fraction: float = 0.02


@dataclass
class StatsModelResult:
    sleep_scores: np.ndarray
    confidence: np.ndarray
    wake_bouts: list[tuple[float, float]]
    rem_bouts: list[tuple[float, float]]
    low_band_means: np.ndarray
    column_times: np.ndarray
    normalization_range: tuple[float, float]
    rem_diagnostics: list[dict[str, float | bool]]
    rem_recovery_diagnostics: list[dict[str, float | bool]]


def sleep_stage_values() -> tuple[int, int, int]:
    """Return Wake, NREM, and REM stage values following make_figure_dev order."""
    return (
        STAGE_NAMES.index("Wake: 1"),
        STAGE_NAMES.index("NREM: 2"),
        STAGE_NAMES.index("REM: 3"),
    )


def _as_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(np.asarray(value).item())


def eeg_time_range(mat: dict) -> tuple[float, float]:
    eeg = np.asarray(mat["eeg"]).flatten()
    eeg_frequency = _as_scalar(mat["eeg_frequency"])
    start_time = _as_scalar(mat.get("start_time"), default=0.0)
    duration = math.ceil((eeg.size - 1) / eeg_frequency)
    return start_time, start_time + duration


def normalize_spectrogram_values(
    z: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, float, float]:
    """Clip spectrogram values by percentile, then scale to 0..1."""
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
    config: StatsModelConfig,
) -> tuple[go.Heatmap, go.Scatter, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Compute the app-style low-band feature used for Wake detection."""
    eeg = np.asarray(mat["eeg"]).flatten()
    eeg_frequency = _as_scalar(mat["eeg_frequency"])
    start_time = _as_scalar(mat.get("start_time"), default=0.0)

    spectrogram, theta_delta_ratio = get_fft_plots(
        eeg,
        eeg_frequency,
        start_time,
    )

    times = np.asarray(spectrogram.x, dtype=float)
    frequencies = np.asarray(spectrogram.y, dtype=float)
    z = np.asarray(spectrogram.z, dtype=float)
    normalized_z, normalization_min, normalization_max = normalize_spectrogram_values(
        z,
        lower_percentile=config.spectrogram_normalization_range[0],
        upper_percentile=config.spectrogram_normalization_range[1],
    )

    low_hz, high_hz = config.spectrogram_sleep_wave_range
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        raise ValueError(
            f"No spectrogram frequency bins found in {low_hz:g}-"
            f"{high_hz:g} Hz. Available range is "
            f"{frequencies.min():g}-{frequencies.max():g} Hz."
        )

    low_band_means = np.nanmean(normalized_z[band_mask, :], axis=0)
    return (
        spectrogram,
        theta_delta_ratio,
        times,
        frequencies[band_mask],
        low_band_means,
        (normalization_min, normalization_max),
    )


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
    wake_merge_coefficient: float,
) -> np.ndarray:
    """Convert NREM gaps to Wake when small relative to neighboring Wake."""
    if not 0 <= wake_merge_coefficient:
        raise ValueError(
            "Wake merge coefficient must be non-negative, "
            f"got {wake_merge_coefficient:g}."
        )

    wake_columns = np.asarray(wake_columns, dtype=bool).copy()
    if wake_columns.size == 0 or wake_merge_coefficient == 0:
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
    merge_runs &= durations < wake_merge_coefficient * neighboring_wake_duration

    for start, end in zip(starts[merge_runs], ends[merge_runs]):
        wake_columns[start:end] = True

    return wake_columns


def remove_short_wake_bouts(
    bouts: list[tuple[float, float]],
    min_wake_duration: float,
) -> list[tuple[float, float]]:
    if min_wake_duration <= 0:
        return bouts
    return [
        (start, end)
        for start, end in bouts
        if end - start >= min_wake_duration
    ]


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


def classify_rem_bouts_from_wake_bouts(
    wake_bouts: list[tuple[float, float]],
    ne: np.ndarray | None,
    time_ne: np.ndarray | None,
    config: StatsModelConfig,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[dict[str, float | bool]]]:
    """Relabel Wake bouts as REM when NE is globally low and valley-shaped."""
    if ne is None or time_ne is None or ne.size == 0 or time_ne.size != ne.size:
        return wake_bouts, [], []

    finite_ne = ne[np.isfinite(ne)]
    if finite_ne.size == 0:
        return wake_bouts, [], []

    global_low_threshold = float(np.nanpercentile(finite_ne, config.rem_threshold_percentile))
    remaining_wake_bouts = []
    rem_bouts = []
    diagnostics: list[dict[str, float | bool]] = []

    for bout_start, bout_end in wake_bouts:
        duration_s = bout_end - bout_start
        in_bout = (time_ne >= bout_start) & (time_ne < bout_end)
        ne_segment = ne[in_bout]
        has_segment = ne_segment.size > 0 and np.any(np.isfinite(ne_segment))
        low_ne_value = (
            float(np.nanpercentile(ne_segment, config.rem_threshold_comparison_percentile))
            if has_segment
            else np.nan
        )
        reaches_global_low = bool(has_segment and low_ne_value <= global_low_threshold)
        # Shape gating is intentionally disabled here to match the working
        # "shape_test=none" comparison pipeline used during validation.
        is_rem = bool(
            duration_s >= config.min_rem_duration
            and reaches_global_low
        )

        diagnostics.append(
            {
                "start": float(bout_start),
                "end": float(bout_end),
                "duration_s": float(duration_s),
                "low_ne_percentile": float(config.rem_threshold_comparison_percentile),
                "low_ne_value": low_ne_value,
                "global_low_threshold": global_low_threshold,
                "reaches_global_low": reaches_global_low,
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
    config: StatsModelConfig,
) -> tuple[
    list[tuple[float, float]],
    list[tuple[float, float]],
    list[dict[str, float | bool]],
]:
    """Split post-trough NE recovery tails from REM bouts into Wake."""
    if config.rem_recovery_epsilon_fraction < 0:
        raise ValueError(
            "REM recovery epsilon fraction must be non-negative, "
            f"got {config.rem_recovery_epsilon_fraction:g}."
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
                epsilon = (
                    config.rem_recovery_epsilon_fraction * ne_range
                    if np.isfinite(ne_range)
                    else 0.0
                )
                cumulative_diff = np.r_[0.0, np.cumsum(np.diff(segment))]
                recovery_indices = np.flatnonzero(
                    cumulative_diff[trough_local_index:] > epsilon
                )
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


def bouts_to_sleep_scores(
    wake_bouts: list[tuple[float, float]],
    rem_bouts: list[tuple[float, float]],
    start_time: float,
    end_time: float,
) -> np.ndarray:
    wake_stage_value, nrem_stage_value, rem_stage_value = sleep_stage_values()
    duration = math.ceil(end_time - start_time)
    sleep_scores = np.full(duration, nrem_stage_value, dtype=int)

    for bout_start, bout_end in wake_bouts:
        start_index = max(0, math.floor(bout_start - start_time))
        end_index = min(duration, math.ceil(bout_end - start_time))
        sleep_scores[start_index:end_index] = wake_stage_value

    for bout_start, bout_end in rem_bouts:
        start_index = max(0, math.floor(bout_start - start_time))
        end_index = min(duration, math.ceil(bout_end - start_time))
        sleep_scores[start_index:end_index] = rem_stage_value

    return sleep_scores


def predict_stats_model(
    mat: dict,
    config: StatsModelConfig | None = None,
) -> StatsModelResult:
    """Run the statistical model and return sleep scores plus diagnostics."""
    if config is None:
        config = StatsModelConfig()

    start_time, end_time = eeg_time_range(mat)
    (
        _spectrogram,
        _theta_delta_ratio,
        times,
        _band_frequencies,
        low_band_means,
        normalization_range,
    ) = compute_spectrogram_feature(mat, config)

    wake_columns = low_band_means <= config.wake_threshold
    edges = spectrogram_column_edges(times, start_time=start_time, end_time=end_time)
    wake_columns = merge_relative_nrem_gaps_once(
        wake_columns,
        edges,
        wake_merge_coefficient=config.wake_merge_coefficient,
    )
    wake_bouts = wake_columns_to_bouts(wake_columns, edges)
    wake_bouts = remove_short_wake_bouts(
        wake_bouts,
        min_wake_duration=config.min_wake_duration,
    )

    ne = mat.get("ne")
    ne_frequency = mat.get("ne_frequency")
    time_ne = None
    ne_for_rem = None
    if ne is not None and ne_frequency is not None and np.asarray(ne).size > 1:
        ne = np.asarray(ne).flatten()
        ne_frequency = _as_scalar(ne_frequency)
        time_ne = ne_time_axis(ne, ne_frequency, start_time)
        ne_for_rem = moving_average_ne(ne, ne_frequency, config.ne_smoothing_window)

    rem_bouts: list[tuple[float, float]] = []
    rem_diagnostics: list[dict[str, float | bool]] = []
    rem_recovery_diagnostics: list[dict[str, float | bool]] = []
    if ne_for_rem is not None and time_ne is not None:
        wake_bouts, rem_bouts, rem_diagnostics = classify_rem_bouts_from_wake_bouts(
            wake_bouts,
            ne=ne_for_rem,
            time_ne=time_ne,
            config=config,
        )
        rem_bouts, recovery_wake_bouts, rem_recovery_diagnostics = split_rem_bouts_at_ne_recovery(
            rem_bouts,
            ne=ne_for_rem,
            time_ne=time_ne,
            config=config,
        )
        wake_bouts = sorted(wake_bouts + recovery_wake_bouts)

    sleep_scores = bouts_to_sleep_scores(wake_bouts, rem_bouts, start_time, end_time)
    confidence = np.ones_like(sleep_scores, dtype=float)
    return StatsModelResult(
        sleep_scores=sleep_scores,
        confidence=confidence,
        wake_bouts=wake_bouts,
        rem_bouts=rem_bouts,
        low_band_means=low_band_means,
        column_times=times,
        normalization_range=normalization_range,
        rem_diagnostics=rem_diagnostics,
        rem_recovery_diagnostics=rem_recovery_diagnostics,
    )


def infer(
    data: dict,
    model_path: Path | None = None,
    config: StatsModelConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the app inference API: return predictions and confidence."""
    _ = model_path
    result = predict_stats_model(data, config=config)
    return result.sleep_scores, result.confidence


def sleep_score_heatmap_values(
    sleep_scores: np.ndarray,
    start_time: float,
    end_time: float,
) -> list[list[float]]:
    duration = math.ceil(end_time - start_time)
    heatmap_scores = np.full(duration, np.nan)
    score_count = min(duration, sleep_scores.size)
    heatmap_scores[:score_count] = sleep_scores[:score_count]
    return np.expand_dims(heatmap_scores, axis=0).tolist()


def make_sleep_score_trace(
    sleep_scores: np.ndarray,
    start_time: float,
    end_time: float,
    num_class: int,
) -> go.Heatmap:
    return go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,
        z=sleep_score_heatmap_values(sleep_scores, start_time, end_time),
        name="Stats Model",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,
    )


def ne_for_plot(mat: dict, config: StatsModelConfig) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    start_time, _end_time = eeg_time_range(mat)
    ne = mat.get("ne")
    ne_frequency = mat.get("ne_frequency")
    if ne is None or ne_frequency is None or np.asarray(ne).size <= 1:
        return None, None, None

    ne = np.asarray(ne).flatten()
    ne_frequency = _as_scalar(ne_frequency)
    time_ne = ne_time_axis(ne, ne_frequency, start_time)
    smoothed_ne = moving_average_ne(ne, ne_frequency, config.ne_smoothing_window)
    return ne, smoothed_ne, time_ne


def make_figure(
    mat: dict,
    sleep_scores: np.ndarray,
    plot_name: str = "",
    config: StatsModelConfig | None = None,
    default_n_shown_samples: int = 2048,
) -> FigureResampler:
    if config is None:
        config = StatsModelConfig()

    start_time, end_time = eeg_time_range(mat)
    num_class = int(_as_scalar(mat.get("num_class"), default=3.0))
    num_class = max(num_class, 3)
    if num_class not in COLORSCALE:
        num_class = 3

    (
        spectrogram,
        theta_delta_ratio,
        _times,
        _band_frequencies,
        _low_band_means,
        _normalization_range,
    ) = compute_spectrogram_feature(mat, config)
    ne, smoothed_ne, time_ne = ne_for_plot(mat, config)

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

    ne_range = 1.0
    if ne is not None and smoothed_ne is not None and time_ne is not None:
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
                name=f"Smoothed NE ({config.ne_smoothing_window:g}s)",
                line=dict(width=1.5, color="rgb(31, 119, 180)"),
                marker=dict(size=2, color="rgb(31, 119, 180)"),
                showlegend=False,
                mode="lines",
                hovertemplate="<b>time</b>: %{x:.2f}<br><b>smoothed y</b>: %{y}<extra></extra>",
            ),
            hf_x=time_ne,
            hf_y=smoothed_ne,
            row=3,
            col=1,
        )
    else:
        for row_index, text in [(2, "NE unavailable"), (3, "Smoothed NE unavailable")]:
            fig.add_annotation(
                text=text,
                x=0.5,
                xref="paper",
                y=0.5,
                yref=f"y{row_index} domain",
                showarrow=False,
                font=dict(size=12, color="gray"),
            )

    sleep_score_trace = make_sleep_score_trace(
        sleep_scores,
        start_time=start_time,
        end_time=end_time,
        num_class=num_class,
    )
    fig.add_trace(spectrogram, secondary_y=False, row=1, col=1)
    fig.add_trace(theta_delta_ratio, secondary_y=True, row=1, col=1)
    fig.add_trace(sleep_score_trace, row=2, col=1)
    fig.add_trace(sleep_score_trace, row=3, col=1)

    fig.update_layout(
        autosize=True,
        margin=dict(t=70, l=10, r=5, b=25),
        height=820,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=plot_name,
            font=dict(size=16),
            xanchor="left",
            x=0.03,
            automargin=True,
            yref="paper",
        ),
        modebar_remove=["lasso2d", "zoom", "autoScale"],
        dragmode="pan",
        clickmode="event",
        xaxis3=dict(tickformat="digits", tickfont=dict(size=10), automargin=True),
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
        range=[0, 15],
        tickmode="array",
        tickvals=list(range(0, 16, 5)),
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
    return fig


if __name__ == "__main__":
    import plotly.io as io

    from scipy.io import loadmat

    MAT_FILE = REPO_ROOT / "user_test_files" / "35_app13.mat"

    config = StatsModelConfig(
        wake_threshold=0.7,
        spectrogram_sleep_wave_range=(1.0, 7.0),
        spectrogram_normalization_range=(5.0, 95.0),
        min_wake_duration=5.0,
        wake_merge_coefficient=0.5,
        min_rem_duration=30.0,
        rem_threshold_percentile=10.0,
        rem_threshold_comparison_percentile=5.0,
        ne_smoothing_window=10.0,
        rem_recovery_epsilon_fraction=0.02,
    )

    io.renderers.default = "browser"
    mat = loadmat(MAT_FILE, squeeze_me=True)
    result = predict_stats_model(mat, config=config)
    fig = make_figure(
        mat,
        result.sleep_scores,
        plot_name=MAT_FILE.name,
        config=config,
    )
    fig.show_dash(mode="external", config={"scrollZoom": True})
