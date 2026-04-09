"""Tests for ChatGPT helper utilities."""

import pytest

from app_src.chatgpt_tools import get_interval_features
from app_src.make_figure_dev import make_figure


def test_get_interval_features_reads_spectral_data_from_figure(mock_mat_data_with_ne):
    """Spectral summaries should come from the already-rendered figure when provided."""
    mat = {
        **mock_mat_data_with_ne,
        "start_time": 10,
    }
    fig = make_figure(mat, plot_name="test")

    features = get_interval_features(mat, start_s=12, end_s=22, fig=fig)

    assert features["spectral_source"] == "figure"
    assert features["duration_s"] == pytest.approx(10.0)
    assert features["spectrogram_frequency_bin_count"] > 0
    assert features["spectrogram_time_bin_count"] > 0
    assert features["delta_power_mean_db"] is not None
    assert features["theta_power_mean_db"] is not None
    assert features["theta_delta_ratio_mean_db"] is not None
    assert features["emg_rms"] is not None
    assert features["current_score_counts"]["Wake"] == 10
    assert features["current_score_dominant_state"] == "Wake"
    assert features["ne_level_mean"] is not None
    assert features["ne_drop_score"] is not None


def test_get_interval_features_recomputes_spectral_data_without_figure(mock_mat_data):
    """The helper should still work when only raw mat data is available."""
    features_without_fig = get_interval_features(mock_mat_data, start_s=20, end_s=40)
    fig = make_figure(mock_mat_data, plot_name="test")
    features_with_fig = get_interval_features(mock_mat_data, start_s=20, end_s=40, fig=fig)

    assert features_without_fig["spectral_source"] == "recomputed"
    assert sum(features_without_fig["current_score_counts"].values()) == 20
    assert features_without_fig["ne_level_mean"] is None
    assert features_without_fig["delta_power_mean_db"] == pytest.approx(
        features_with_fig["delta_power_mean_db"],
        abs=1e-6,
    )
    assert features_without_fig["theta_power_mean_db"] == pytest.approx(
        features_with_fig["theta_power_mean_db"],
        abs=1e-6,
    )
    assert features_without_fig["theta_delta_ratio_mean_db"] == pytest.approx(
        features_with_fig["theta_delta_ratio_mean_db"],
        abs=1e-6,
    )
