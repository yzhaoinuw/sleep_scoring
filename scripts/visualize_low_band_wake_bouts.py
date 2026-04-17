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
CHATGPT_XAXIS_NTICKS = 24
PLOTLY_CONFIG = {"scrollZoom": True}


def _as_scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(np.asarray(value).item())


def normalize_like_plotly_heatmap(z: np.ndarray) -> np.ndarray:
    """Approximate Plotly's default heatmap color normalization for one trace."""
    z_min = np.nanmin(z)
    z_max = np.nanmax(z)
    z_range = z_max - z_min
    if not np.isfinite(z_range) or z_range == 0:
        return np.zeros_like(z, dtype=float)
    return (z - z_min) / z_range


def compute_spectrogram_feature(
    mat: dict,
    low_hz: float,
    high_hz: float,
    window_duration: float,
) -> tuple[go.Heatmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the app-style spectrogram and normalized low-band column means."""
    eeg = np.asarray(mat["eeg"]).flatten()
    eeg_frequency = _as_scalar(mat["eeg_frequency"])
    start_time = _as_scalar(mat.get("start_time"), default=0.0)

    spectrogram, _ = get_fft_plots(
        eeg,
        eeg_frequency,
        start_time,
        window_duration=window_duration,
    )

    times = np.asarray(spectrogram.x, dtype=float)
    frequencies = np.asarray(spectrogram.y, dtype=float)
    z = np.asarray(spectrogram.z, dtype=float)
    normalized_z = normalize_like_plotly_heatmap(z)

    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        raise ValueError(
            f"No spectrogram frequency bins found in {low_hz:g}-{high_hz:g} Hz. "
            f"Available range is {frequencies.min():g}-{frequencies.max():g} Hz."
        )

    low_band_means = np.nanmean(normalized_z[band_mask, :], axis=0)
    return spectrogram, times, frequencies[band_mask], low_band_means, frequencies


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


def merge_close_bouts(
    bouts: list[tuple[float, float]],
    merge_gap_s: float,
) -> list[tuple[float, float]]:
    if not bouts or merge_gap_s <= 0:
        return bouts

    merged = [bouts[0]]
    for start, end in bouts[1:]:
        previous_start, previous_end = merged[-1]
        if start - previous_end <= merge_gap_s:
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    return merged


def wake_columns_to_bouts(
    wake_columns: np.ndarray,
    edges: np.ndarray,
    min_bout_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
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

    bouts = merge_close_bouts(bouts, merge_gap_s=merge_gap_s)
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
    start_time: float,
    end_time: float,
) -> list[list[float]]:
    """Create a one-row app-style sleep-score heatmap with Wake only."""
    duration = math.ceil(end_time - start_time)
    sleep_scores = np.full(duration, np.nan)
    for bout_start, bout_end in wake_bouts:
        start_index = max(0, math.floor(bout_start - start_time))
        end_index = min(duration, math.ceil(bout_end - start_time))
        sleep_scores[start_index:end_index] = 0
    return np.expand_dims(sleep_scores, axis=0).tolist()


def make_wake_sleep_score_trace(
    wake_bouts: list[tuple[float, float]],
    start_time: float,
    end_time: float,
    num_class: int,
) -> go.Heatmap:
    """Build the Wake-only overlay using the app's sleep-stage colors."""
    return go.Heatmap(
        x0=start_time + 0.5,
        dx=1,
        y0=0,
        dy=HEATMAP_WIDTH,
        z=wake_bouts_to_sleep_score_heatmap(wake_bouts, start_time, end_time),
        name="Threshold Wake",
        hoverinfo="none",
        colorscale=COLORSCALE[num_class],
        showscale=False,
        opacity=SLEEP_SCORE_OPACITY,
        zmax=num_class - 1,
        zmin=0,
        showlegend=False,
        xgap=0.05,
    )


def make_wake_bout_figure(
    mat: dict,
    mat_name: str,
    threshold: float,
    wake_when: str = "below",
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
    min_bout_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
    default_n_shown_samples: int = 2048,
    num_class: int = 3,
) -> tuple[go.Figure, list[tuple[float, float]], np.ndarray]:
    start_time, end_time = eeg_time_range(mat)
    if mat.get("num_class") is not None:
        num_class = int(_as_scalar(mat["num_class"]))
    if num_class not in COLORSCALE:
        num_class = 3

    spectrogram, times, band_frequencies, low_band_means, _frequencies = (
        compute_spectrogram_feature(
            mat,
            low_hz=low_hz,
            high_hz=high_hz,
            window_duration=window_duration,
        )
    )
    wake_columns = threshold_wake_columns(
        low_band_means,
        threshold=threshold,
        wake_when=wake_when,
    )
    edges = spectrogram_column_edges(times, start_time=start_time, end_time=end_time)
    wake_bouts = wake_columns_to_bouts(
        wake_columns,
        edges,
        min_bout_duration_s=min_bout_duration_s,
        merge_gap_s=merge_gap_s,
    )

    fig = FigureResampler(
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("EEG Spectrogram", "NE"),
            row_heights=[0.45, 0.55],
            specs=[[{"r": -0.05}], [{"r": -0.05}]],
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
    if ne is not None and ne_frequency is not None and np.asarray(ne).size > 1:
        ne = np.asarray(ne).flatten()
        ne_frequency = _as_scalar(ne_frequency)
        ne_end_time = (ne.size - 1) / ne_frequency + start_time
        time_ne = np.linspace(start_time, ne_end_time, ne.size)
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

    wake_sleep_scores = make_wake_sleep_score_trace(
        wake_bouts,
        start_time=start_time,
        end_time=end_time,
        num_class=num_class,
    )

    fig.add_trace(spectrogram, row=1, col=1)
    fig.add_trace(wake_sleep_scores, row=2, col=1)

    total_wake_s = sum(end - start for start, end in wake_bouts)
    fig.update_layout(
        autosize=True,
        margin=dict(t=70, l=10, r=5, b=25),
        height=700,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255, 255, 255, 0.6)"),
        title=dict(
            text=(
                f"{mat_name}: Wake-only low-band threshold view"
                f"<br><sup>{low_hz:g}-{high_hz:g} Hz mean {wake_when} "
                f"{threshold:g}; {len(wake_bouts)} bouts; {total_wake_s:.1f} s total</sup>"
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
        xaxis2=dict(
            tickformat="digits",
            nticks=CHATGPT_XAXIS_NTICKS,
            tickfont=dict(size=10),
            automargin=True,
        ),
    )
    fig.update_traces(xaxis="x2")
    fig.update_xaxes(range=[start_time, end_time], row=1, col=1)
    fig.update_xaxes(
        range=[start_time, end_time],
        row=2,
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
    fig.update_annotations(font_size=14)
    if fig["layout"]["annotations"]:
        fig["layout"]["annotations"][-1]["font"]["size"] = 14

    print(
        f"Used {band_frequencies.size} frequency bins from "
        f"{band_frequencies.min():.3g}-{band_frequencies.max():.3g} Hz."
    )
    print(f"Wake overlay color: {STAGE_COLORS[0]}")
    return fig, wake_bouts, low_band_means


def format_threshold_for_filename(threshold: float) -> str:
    threshold_text = f"{threshold:g}".replace("-", "neg_").replace(".", "p")
    return threshold_text


def default_output_path(
    mat_path: Path,
    output_dir: Path | None,
    threshold: float,
    wake_when: str,
) -> Path:
    base_dir = output_dir if output_dir is not None else mat_path.parent
    threshold_text = format_threshold_for_filename(threshold)
    return base_dir / (
        f"{mat_path.stem}_low_band_wake_bouts_" f"{wake_when}_threshold_{threshold_text}.html"
    )


def run_wake_bout_visualization(
    mat_file: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    threshold: float = 0.8,
    wake_when: str = "below",
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
    min_bout_duration_s: float = 0.0,
    merge_gap_s: float = 0.0,
    default_n_shown_samples: int = 2048,
    show: bool = False,
) -> Path:
    mat_path = mat_file.expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(mat_path, squeeze_me=True)
    fig, wake_bouts, low_band_means = make_wake_bout_figure(
        mat,
        mat_name=mat_path.name,
        threshold=threshold,
        wake_when=wake_when,
        low_hz=low_hz,
        high_hz=high_hz,
        window_duration=window_duration,
        min_bout_duration_s=min_bout_duration_s,
        merge_gap_s=merge_gap_s,
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
        ).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, config=PLOTLY_CONFIG)

    quantiles = np.nanquantile(low_band_means, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    print(f"Wrote {output_path}")
    print(
        f"Wake rule: normalized {low_hz:g}-{high_hz:g} Hz column mean " f"{wake_when} {threshold:g}"
    )
    print(f"Predicted Wake bouts: {len(wake_bouts)}")
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
    parser.add_argument("--min-bout-duration-s", type=float, default=0.0)
    parser.add_argument("--merge-gap-s", type=float, default=0.0)
    parser.add_argument("--default-n-shown-samples", type=int, default=2048)
    parser.add_argument("--show", action="store_true", help="Open the figure after writing HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_wake_bout_visualization(
        mat_file=args.mat_file,
        output=args.output,
        output_dir=args.output_dir,
        threshold=args.threshold,
        wake_when=args.wake_when,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        window_duration=args.window_duration,
        min_bout_duration_s=args.min_bout_duration_s,
        merge_gap_s=args.merge_gap_s,
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
        THRESHOLD = 0.6
        WAKE_WHEN = "below"  # "below" means low 1-5 Hz power marks Wake.
        LOW_HZ = 1.0
        HIGH_HZ = 5.0
        WINDOW_DURATION = 5.0
        MIN_BOUT_DURATION_S = 0.0
        MERGE_GAP_S = 0.0
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
            min_bout_duration_s=MIN_BOUT_DURATION_S,
            merge_gap_s=MERGE_GAP_S,
            default_n_shown_samples=DEFAULT_N_SHOWN_SAMPLES,
            show=SHOW,
        )
