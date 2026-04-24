"""Plot the distribution of normalized low-band spectrogram column means.

This is an exploratory helper for the statistical wake-detection experiment.
It reuses the app's spectrogram builder so the feature is derived from the
same smoothed dB heatmap shown in the UI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app_src.get_fft_plots import get_fft_plots  # noqa: E402


STAGE_NAMES = {
    0: "Wake",
    1: "NREM",
    2: "REM",
    3: "MA",
}


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


def compute_low_band_column_means(
    mat: dict,
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return spectrogram times, frequencies, and normalized low-band means."""
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

    column_means = np.nanmean(normalized_z[band_mask, :], axis=0)
    return times, frequencies[band_mask], column_means


def stage_labels_for_times(mat: dict, times: np.ndarray) -> np.ndarray | None:
    sleep_scores = mat.get("sleep_scores")
    if sleep_scores is None:
        return None

    sleep_scores = np.asarray(sleep_scores).flatten()
    if sleep_scores.size == 0:
        return None

    start_time = _as_scalar(mat.get("start_time"), default=0.0)
    score_indices = np.floor(times - start_time).astype(int)
    valid = (score_indices >= 0) & (score_indices < sleep_scores.size)
    labels = np.full(times.shape, np.nan)
    labels[valid] = sleep_scores[score_indices[valid]]
    labels[labels < 0] = np.nan
    return labels


def build_distribution_figure(
    mat_name: str,
    values: np.ndarray,
    times: np.ndarray,
    labels: np.ndarray | None,
    low_hz: float,
    high_hz: float,
) -> go.Figure:
    has_labels = labels is not None and np.any(np.isfinite(labels))
    rows = 2 if has_labels else 1
    subplot_titles = ["All spectrogram columns"]
    if has_labels:
        subplot_titles.append("Columns grouped by sleep score")

    fig = make_subplots(rows=rows, cols=1, subplot_titles=subplot_titles)
    band_label = f"Normalized {low_hz:g}-{high_hz:g} Hz column mean"

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=80,
            name="All columns",
            marker_color="#4C78A8",
            opacity=0.85,
        ),
        row=1,
        col=1,
    )

    if has_labels:
        for stage_value in sorted(np.unique(labels[np.isfinite(labels)]).astype(int)):
            stage_mask = labels == stage_value
            fig.add_trace(
                go.Histogram(
                    x=values[stage_mask],
                    nbinsx=80,
                    name=STAGE_NAMES.get(stage_value, f"Score {stage_value}"),
                    opacity=0.65,
                ),
                row=2,
                col=1,
            )
        fig.update_layout(barmode="overlay")

    fig.update_xaxes(title_text=band_label, range=[0, 1], row=rows, col=1)
    for row in range(1, rows + 1):
        fig.update_yaxes(title_text="Column count", row=row, col=1)

    fig.update_layout(
        title=(
            f"{mat_name}: distribution of normalized spectrogram column means"
            f"<br><sup>{len(values):,} columns, time range "
            f"{np.nanmin(times):.1f}-{np.nanmax(times):.1f} s</sup>"
        ),
        template="plotly_white",
        bargap=0.02,
        height=650 if rows == 2 else 450,
        legend_title_text="Group",
    )
    return fig


def default_output_path(mat_path: Path, output_dir: Path | None) -> Path:
    base_dir = output_dir if output_dir is not None else mat_path.parent
    return base_dir / f"{mat_path.stem}_low_band_mean_distribution.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the normalized 1-5 Hz spectrogram column mean for a .mat file "
            "and plot its distribution."
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
    parser.add_argument("--low-hz", type=float, default=1.0, help="Low band edge in Hz.")
    parser.add_argument("--high-hz", type=float, default=5.0, help="High band edge in Hz.")
    parser.add_argument(
        "--window-duration",
        type=float,
        default=5.0,
        help="Spectrogram window duration in seconds, matching get_fft_plots default.",
    )
    return parser.parse_args()


def run_distribution_plot(
    mat_file: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    low_hz: float = 1.0,
    high_hz: float = 5.0,
    window_duration: float = 5.0,
) -> Path:
    mat_path = mat_file.expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    mat = loadmat(mat_path, squeeze_me=True)
    times, frequencies, values = compute_low_band_column_means(
        mat,
        low_hz=low_hz,
        high_hz=high_hz,
        window_duration=window_duration,
    )
    labels = stage_labels_for_times(mat, times)

    output_path = (
        output.expanduser().resolve()
        if output is not None
        else default_output_path(mat_path, output_dir).resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_distribution_figure(
        mat_path.name,
        values,
        times,
        labels,
        low_hz,
        high_hz,
    )
    fig.write_html(output_path)

    quantiles = np.nanquantile(values, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    print(f"Wrote {output_path}")
    print(
        f"Used {frequencies.size} frequency bins from "
        f"{frequencies.min():.3g}-{frequencies.max():.3g} Hz."
    )
    print(
        "Column mean quantiles "
        "(min, p10, p25, median, p75, p90, max): "
        + ", ".join(f"{value:.4f}" for value in quantiles)
    )
    return output_path


def main() -> None:
    args = parse_args()
    run_distribution_plot(
        mat_file=args.mat_file,
        output=args.output,
        output_dir=args.output_dir,
        low_hz=args.low_hz,
        high_hz=args.high_hz,
        window_duration=args.window_duration,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Editable parameters for direct console/Spyder-style runs.
        MAT_FILE = REPO_ROOT / "user_test_files" / "408_yfp.mat"
        OUTPUT = None  # Path to a specific .html file, or None for the default name.
        OUTPUT_DIR = REPO_ROOT / "test-artifacts"
        LOW_HZ = 1.0
        HIGH_HZ = 5.0
        WINDOW_DURATION = 5.0

        run_distribution_plot(
            mat_file=MAT_FILE,
            output=OUTPUT,
            output_dir=OUTPUT_DIR,
            low_hz=LOW_HZ,
            high_hz=HIGH_HZ,
            window_duration=WINDOW_DURATION,
        )
