"""Tests for app_src/get_fft_plots.py"""

import numpy as np
import plotly.graph_objects as go

from app_src.get_fft_plots import get_fft_plots


class TestGetFFTPlots:
    """Tests for get_fft_plots function."""

    def test_returns_correct_types(self, mock_eeg_signal):
        """Test that function returns correct plotly object types."""
        spectrogram, theta_delta = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        assert isinstance(spectrogram, go.Heatmap)
        assert isinstance(theta_delta, go.Scatter)

    def test_spectrogram_has_data(self, mock_eeg_signal):
        """Test that spectrogram contains valid data."""
        spectrogram, _ = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        # z should contain the power spectral density data
        assert spectrogram.z is not None
        assert len(spectrogram.z) > 0

        # x should be time axis
        assert spectrogram.x is not None
        assert len(spectrogram.x) > 0

        # y should be frequency axis
        assert spectrogram.y is not None
        assert len(spectrogram.y) > 0

    def test_frequency_range_limited_to_30hz(self, mock_eeg_signal):
        """Test that frequency axis is limited to 30 Hz."""
        spectrogram, _ = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        # All frequencies should be <= 30 Hz
        assert max(spectrogram.y) <= 30

    def test_theta_delta_ratio_has_data(self, mock_eeg_signal):
        """Test that theta/delta ratio trace contains valid data."""
        _, theta_delta = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        assert theta_delta.x is not None
        assert theta_delta.y is not None
        assert len(theta_delta.x) == len(theta_delta.y)

    def test_start_time_offset(self, mock_eeg_signal):
        """Test that start_time offsets the time axis correctly."""
        start_time = 100

        spectrogram, theta_delta = get_fft_plots(
            mock_eeg_signal, eeg_frequency=512, start_time=start_time
        )

        # Time axis should start at or after start_time
        assert min(spectrogram.x) >= start_time
        assert min(theta_delta.x) >= start_time

    def test_fractional_sample_rate_time_axis_does_not_drift(self):
        """Test that fractional sampling rates stay anchored to exact time bins."""
        eeg_frequency = 512.25
        recording_duration = 120
        signal = np.random.randn(round(eeg_frequency * recording_duration)).astype(np.float32)

        spectrogram, theta_delta = get_fft_plots(signal, eeg_frequency=eeg_frequency, start_time=0)

        expected_times = np.arange(len(spectrogram.x)) * 2.5
        assert np.allclose(spectrogram.x, expected_times)
        assert np.allclose(theta_delta.x, expected_times)

    def test_time_axis_includes_final_overlapping_bin(self):
        """Test that the final padded bin is included for non-grid-aligned recordings."""
        eeg_frequency = 512.25
        signal = np.random.randn(round(eeg_frequency * 121)).astype(np.float32)

        spectrogram, _ = get_fft_plots(signal, eeg_frequency=eeg_frequency, start_time=0)

        recording_duration = signal.size / eeg_frequency
        expected_last_time = np.ceil(recording_duration / 2.5) * 2.5
        assert np.isclose(spectrogram.x[-1], expected_last_time)

    def test_different_window_duration(self, mock_eeg_signal):
        """Test with different window duration."""
        spec_5s, _ = get_fft_plots(
            mock_eeg_signal, eeg_frequency=512, start_time=0, window_duration=5
        )
        spec_10s, _ = get_fft_plots(
            mock_eeg_signal, eeg_frequency=512, start_time=0, window_duration=10
        )

        # Different window sizes should produce different frequency resolutions
        # (longer window = more frequency bins)
        assert spec_5s.y is not None
        assert spec_10s.y is not None

    def test_spectrogram_colorscale_set(self, mock_eeg_signal):
        """Test that spectrogram uses the configured colorscale."""
        spectrogram, _ = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        # Should have colorscale configured (from config)
        assert spectrogram.colorscale is not None

    def test_theta_delta_line_properties(self, mock_eeg_signal):
        """Test theta/delta trace line properties."""
        _, theta_delta = get_fft_plots(mock_eeg_signal, eeg_frequency=512, start_time=0)

        assert theta_delta.mode == "lines"
        assert theta_delta.showlegend is False

    def test_short_signal(self):
        """Test with a short signal (edge case)."""
        # 10 seconds of data
        short_signal = np.random.randn(512 * 10).astype(np.float32)

        spectrogram, theta_delta = get_fft_plots(short_signal, eeg_frequency=512, start_time=0)

        assert spectrogram.z is not None
        assert theta_delta.y is not None
