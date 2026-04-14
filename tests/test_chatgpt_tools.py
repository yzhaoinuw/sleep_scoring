"""Tests for ChatGPT helper utilities."""

import numpy as np
import pytest

from app_src.chatgpt_tools import (
    _get_bottom_xaxis_layout_key,
    apply_transition_rules,
    get_current_scores,
    get_interval_features,
    mark_uncertain_interval,
    set_scores_block,
)
from app_src.make_figure_chatgpt import make_chatgpt_vision_figure
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


def test_make_figure_uses_denser_overview_ticks_without_minor_hour_marks(mock_mat_data):
    """The active figure should request denser x labels and disable minor hour ticks."""
    fig = make_figure(mock_mat_data, plot_name="test")

    assert fig.layout.xaxis4.tickformat == "digits"
    assert fig.layout.xaxis4.nticks == 16
    assert fig.layout.xaxis4.minor.to_plotly_json() == {}
    assert tuple(fig.layout.yaxis.range) == (0, 30)
    assert tuple(fig.layout.yaxis.tickvals) == (0, 10, 20, 30)
    assert fig.layout.yaxis2.showgrid is False


def test_make_chatgpt_vision_figure_only_shows_spectrogram_and_ne(mock_mat_data_with_ne):
    """The model-facing export figure should omit the raw EEG and EMG waveform panels."""
    fig = make_chatgpt_vision_figure(mock_mat_data_with_ne, plot_name="test")

    trace_names = [getattr(trace, "name", None) for trace in fig.data]

    assert "Spectrogram" in trace_names
    assert "Theta/Delta" in trace_names
    assert "NE" in trace_names
    assert "EEG" not in trace_names
    assert "EMG" not in trace_names
    assert fig.layout.xaxis2.tickformat == "digits"
    assert fig.layout.xaxis2.nticks == 16
    assert tuple(fig.layout.yaxis.range) == (0, 15)
    assert tuple(fig.layout.yaxis.tickvals) == (0, 5, 10, 15)
    assert fig.layout.yaxis2.showgrid is False


def test_make_chatgpt_vision_figure_marks_missing_ne(mock_mat_data):
    """Recordings without NE should still export a focused figure with a clear note."""
    fig = make_chatgpt_vision_figure(mock_mat_data, plot_name="test")

    annotation_text = [annotation.text for annotation in fig.layout.annotations]

    assert "NE unavailable" in annotation_text


def test_zoom_snapshot_helper_detects_bottom_xaxis_for_both_figure_layouts(
    mock_mat_data,
    mock_mat_data_with_ne,
):
    """Zoom export should work with both the 4-row UI figure and the 2-row ChatGPT figure."""
    full_figure = make_figure(mock_mat_data, plot_name="full")
    chatgpt_figure = make_chatgpt_vision_figure(mock_mat_data_with_ne, plot_name="chatgpt")

    assert _get_bottom_xaxis_layout_key(full_figure) == "xaxis4"
    assert _get_bottom_xaxis_layout_key(chatgpt_figure) == "xaxis2"


def test_get_current_scores_returns_raw_scores_and_contiguous_blocks(sample_sleep_scores):
    """The helper should expose both per-second labels and merged blocks."""
    current_scores = get_current_scores(sample_sleep_scores, start_s=1.2, end_s=7.1)

    assert current_scores["start_s"] == 1
    assert current_scores["end_s"] == 8
    assert current_scores["duration_s"] == 7
    assert current_scores["current_score_counts"] == {
        "Wake": 1,
        "NREM": 4,
        "REM": 2,
        "Unscored": 0,
    }
    assert current_scores["current_score_dominant_state"] == "NREM"
    assert current_scores["scores"] == [
        {"second": 1, "state": "Wake", "score": 0},
        {"second": 2, "state": "NREM", "score": 1},
        {"second": 3, "state": "NREM", "score": 1},
        {"second": 4, "state": "NREM", "score": 1},
        {"second": 5, "state": "REM", "score": 2},
        {"second": 6, "state": "REM", "score": 2},
        {"second": 7, "state": "NREM", "score": 1},
    ]
    assert current_scores["score_blocks"] == [
        {"start_s": 1, "end_s": 2, "duration_s": 1, "state": "Wake", "score": 0},
        {"start_s": 2, "end_s": 5, "duration_s": 3, "state": "NREM", "score": 1},
        {"start_s": 5, "end_s": 7, "duration_s": 2, "state": "REM", "score": 2},
        {"start_s": 7, "end_s": 8, "duration_s": 1, "state": "NREM", "score": 1},
    ]


def test_get_current_scores_clamps_and_preserves_unscored_values(sleep_scores_with_missing):
    """Out-of-range requests should clamp and keep missing labels explicit."""
    current_scores = get_current_scores(sleep_scores_with_missing, start_s=-2, end_s=20)

    assert current_scores["start_s"] == 0
    assert current_scores["end_s"] == 9
    assert current_scores["duration_s"] == 9
    assert current_scores["current_score_counts"] == {
        "Wake": 2,
        "NREM": 2,
        "REM": 1,
        "Unscored": 4,
    }
    assert current_scores["current_score_dominant_state"] == "Wake"
    assert current_scores["scores"][0] == {"second": 0, "state": "Unscored", "score": None}
    assert current_scores["scores"][-1] == {"second": 8, "state": "Unscored", "score": None}
    assert current_scores["score_blocks"] == [
        {"start_s": 0, "end_s": 2, "duration_s": 2, "state": "Unscored", "score": None},
        {"start_s": 2, "end_s": 3, "duration_s": 1, "state": "Wake", "score": 0},
        {"start_s": 3, "end_s": 5, "duration_s": 2, "state": "NREM", "score": 1},
        {"start_s": 5, "end_s": 6, "duration_s": 1, "state": "REM", "score": 2},
        {"start_s": 6, "end_s": 7, "duration_s": 1, "state": "Wake", "score": 0},
        {"start_s": 7, "end_s": 9, "duration_s": 2, "state": "Unscored", "score": None},
    ]


def test_get_current_scores_validates_interval_inputs():
    """Invalid score requests should fail with a clear error."""
    with pytest.raises(ValueError, match="end_s must be greater than start_s"):
        get_current_scores(np.array([0, 1, 2]), start_s=4, end_s=4)

    with pytest.raises(ValueError, match="Requested interval falls outside the available scores"):
        get_current_scores(np.array([0, 1, 2]), start_s=3, end_s=5)


def test_set_scores_block_updates_a_copied_half_open_interval(sample_sleep_scores):
    """Score writeback should follow the app's existing half-open interval behavior."""
    updated_scores = set_scores_block(sample_sleep_scores, start_s=1.2, end_s=5.0, state="REM")

    np.testing.assert_array_equal(sample_sleep_scores, np.array([0, 0, 1, 1, 1, 2, 2, 1, 0]))
    np.testing.assert_array_equal(updated_scores, np.array([0, 2, 2, 2, 2, 2, 2, 1, 0]))


def test_set_scores_block_clamps_to_available_score_range(sample_sleep_scores):
    """Out-of-range edits should clamp to the array bounds instead of failing."""
    updated_scores = set_scores_block(sample_sleep_scores, start_s=-5, end_s=20, state="Wake")

    np.testing.assert_array_equal(updated_scores, np.zeros(sample_sleep_scores.shape, dtype=float))


def test_set_scores_block_validates_inputs():
    """Invalid edit requests should raise clear errors."""
    with pytest.raises(ValueError, match="end_s must be greater than start_s"):
        set_scores_block(np.array([0, 1, 2]), start_s=2, end_s=2, state="Wake")

    with pytest.raises(ValueError, match="Requested interval falls outside the available scores"):
        set_scores_block(np.array([0, 1, 2]), start_s=5, end_s=6, state="Wake")

    with pytest.raises(ValueError, match="Unsupported sleep state"):
        set_scores_block(np.array([0, 1, 2]), start_s=0, end_s=1, state="MA")


def test_apply_transition_rules_preserves_scores_without_local_rewrites():
    """Transition rules are prompt guidance, so the helper should not mutate scores."""
    scores = np.array([0, 0, 2, 2, 1, 1], dtype=float)

    updated_scores = apply_transition_rules(scores)

    np.testing.assert_array_equal(scores, np.array([0, 0, 2, 2, 1, 1], dtype=float))
    np.testing.assert_array_equal(updated_scores, scores)


def test_apply_transition_rules_preserves_unscored_boundaries():
    """The non-destructive helper should preserve missing labels exactly."""
    scores = np.array([0, 0, np.nan, 2, 2, 1, 1], dtype=float)

    updated_scores = apply_transition_rules(scores)

    assert np.array_equal(updated_scores, scores, equal_nan=True)


def test_apply_transition_rules_validates_nonempty_input():
    """Empty score arrays should fail clearly."""
    with pytest.raises(ValueError, match="scores must contain at least one value"):
        apply_transition_rules(np.array([]))


def test_mark_uncertain_interval_normalizes_bounds_and_reason():
    """Uncertain interval markers should use whole-second bounds and trimmed reasons."""
    marker = mark_uncertain_interval(start_s=10.2, end_s=15.0, reason="  transition ambiguity  ")

    assert marker == {
        "start_s": 10,
        "end_s": 15,
        "duration_s": 5,
        "reason": "transition ambiguity",
    }


def test_mark_uncertain_interval_validates_inputs():
    """Invalid uncertain-interval requests should fail clearly."""
    with pytest.raises(ValueError, match="end_s must be greater than start_s"):
        mark_uncertain_interval(start_s=5, end_s=5, reason="ambiguous")

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        mark_uncertain_interval(start_s=1, end_s=2, reason="   ")
