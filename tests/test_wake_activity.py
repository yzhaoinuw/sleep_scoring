from unittest.mock import patch

import numpy as np
import pytest

from app_src.wake_activity import (
    WakeActivityConfig,
    active_mask,
    emg_envelope,
    emg_second_rms,
    nrem_baseline_threshold,
    second_activity,
    subdivide_wake,
)


def test_duration_boundary_and_interior_gap():
    config = WakeActivityConfig(threshold=2, min_duration=5)
    envelope = np.r_[np.ones(20), np.full(100, 5), np.ones(20)]
    active = active_mask(envelope, np.ones(7, bool), 2, config)
    assert active.sum() == 100  # Exactly five seconds qualifies.
    envelope[119] = 1
    assert not active_mask(envelope, np.ones(7, bool), 2, config).any()
    envelope[119] = 5
    envelope[55:65] = 1  # Half-second dip is bridged.
    assert active_mask(envelope, np.ones(7, bool), 2, config).sum() == 100


def test_activity_never_bridges_sleep_or_invalid_bins():
    config = WakeActivityConfig(min_duration=2, gap_tolerance=1)
    envelope = np.full(140, 5.0)
    wake = np.array([True, True, True, False, True, True, True])
    active = active_mask(envelope, wake, 2, config)
    assert not active[60:80].any()
    envelope[25] = np.nan
    active = active_mask(envelope, wake, 2, config)
    assert not active[:60].any()  # The two fragments are both under two seconds.


def test_epoch_labels_use_majority_time_and_partial_last_second():
    active = np.r_[
        np.ones(10, bool), np.zeros(10, bool), np.zeros(19, bool), [True], np.ones(3, bool)
    ]
    np.testing.assert_array_equal(second_activity(active, 3), [True, False, True])


def test_generic_wake_is_not_a_quiet_example_and_coarse_labels_are_preserved():
    coarse = np.array([0] * 6 + [1, 2, 3], dtype=float)
    users = [0] + [None] * 8
    envelope = np.full(180, 4.0)
    with patch("app_src.wake_activity.emg_envelope", return_value=envelope):
        result = subdivide_wake(
            [], 512, coarse, users, WakeActivityConfig(method="nrem_envelope", threshold=2)
        )
    np.testing.assert_array_equal(result.sleep_scores, [4] * 6 + [1, 2, 3])
    assert result.calibrated_seconds == 0
    np.testing.assert_array_equal(coarse, [0] * 6 + [1, 2, 3])


def test_fine_examples_calibrate_before_override_and_force_coarse_wake():
    users = [5] + [None] * 4 + [4] + [None] * 4
    envelope = np.r_[np.ones(100), np.full(100, 6.0)]
    with patch("app_src.wake_activity.emg_envelope", return_value=envelope):
        result = subdivide_wake(
            [],
            512,
            [1] + [0] * 4 + [2] + [0] * 4,
            users,
            WakeActivityConfig(method="nrem_envelope", threshold=100),
        )
    assert 1 <= result.threshold < 6  # Cannot pass by merely overlaying the examples.
    assert result.calibrated_seconds == 2
    np.testing.assert_array_equal(result.sleep_scores, [5] * 5 + [4] * 5)


def test_automatic_threshold_uses_upper_nrem_rms_distribution():
    config = WakeActivityConfig(nrem_baseline_percentile=75, nrem_deviation_multiplier=2)
    envelope = np.r_[np.full(20, 1.0), np.full(20, 2.0), np.full(20, 3.0), np.full(20, 4.0)]
    threshold, percentile, robust_sd, samples = nrem_baseline_threshold(
        envelope, np.ones(4, bool), config
    )
    assert percentile == pytest.approx(3.25)
    assert robust_sd == pytest.approx(1.4826)
    assert threshold == pytest.approx(3.25 + 2 * 1.4826)
    assert samples == 80


def test_automatic_threshold_requires_valid_nrem_emg():
    with patch("app_src.wake_activity.emg_envelope", return_value=np.full(40, 2.0)):
        with pytest.raises(ValueError, match="valid NREM EMG"):
            subdivide_wake([], 512, [0, 0], config=WakeActivityConfig(method="nrem_envelope"))


def test_manual_subtype_shorter_than_minimum_remains_authoritative():
    with patch("app_src.wake_activity.emg_envelope", return_value=np.ones(40)):
        result = subdivide_wake(
            [],
            512,
            [0, 0],
            [4, 5],
            WakeActivityConfig(method="nrem_envelope", threshold=2, min_duration=2),
        )
    np.testing.assert_array_equal(result.sleep_scores, [4, 5])
    assert result.active_bouts == []


def test_missing_emg_fails_in_wake_instead_of_becoming_quiet():
    with patch("app_src.wake_activity.emg_envelope", return_value=np.r_[np.ones(20), np.nan]):
        with pytest.raises(ValueError, match="invalid EMG"):
            subdivide_wake([], 512, [0, 0], config=WakeActivityConfig(method="nrem_envelope"))


def test_invalid_emg_during_sleep_does_not_block_valid_wake():
    with patch(
        "app_src.wake_activity.emg_envelope", return_value=np.r_[np.ones(120), [np.nan] * 20]
    ):
        result = subdivide_wake(
            [], 512, [0] * 6 + [1], config=WakeActivityConfig(method="nrem_envelope", threshold=2)
        )
    np.testing.assert_array_equal(result.sleep_scores, [5] * 6 + [1])


def test_rms_removes_offset_and_linear_drift_without_mutating_emg():
    fs = 512
    times = np.arange(fs * 10) / fs
    clean = 2 * np.sin(2 * np.pi * 80 * times)
    raw = clean + 20 + 0.2 * times
    original = raw.copy()
    envelope = emg_envelope(raw, fs)
    assert envelope.size == 200
    np.testing.assert_allclose(envelope[20:-20], np.sqrt(2), atol=0.015)
    np.testing.assert_array_equal(raw, original)


def test_per_second_rms_removes_offset_and_linear_drift_without_mutating_emg():
    fs = 512
    times = np.arange(fs * 10) / fs
    clean = 2 * np.sin(2 * np.pi * 80 * times)
    raw = clean + 20 + 0.2 * times
    original = raw.copy()
    rms = emg_second_rms(raw, fs, 10)
    np.testing.assert_allclose(rms[1:-1], np.sqrt(2), atol=0.015)
    np.testing.assert_array_equal(raw, original)


def test_per_second_rms_allows_a_partial_final_score_epoch():
    fs = 512.001
    raw = np.sin(2 * np.pi * 60 * np.arange(round(2.2 * fs)) / fs)
    rms = emg_second_rms(raw, fs, 3)
    assert np.all(np.isfinite(rms))

    result = subdivide_wake(raw, fs, [0, 0, 0])
    assert np.all(np.isin(result.sleep_scores, [4, 5]))


def test_per_second_rms_still_rejects_a_wholly_missing_score_epoch():
    fs = 512.001
    raw = np.sin(2 * np.pi * 60 * np.arange(round(2.2 * fs)) / fs)
    with pytest.raises(ValueError, match="invalid EMG"):
        subdivide_wake(raw, fs, [0, 0, 0, 0])


def test_flatline_and_low_sampling_rate_are_not_silent_quiet_predictions():
    with pytest.raises(ValueError, match="flatlined"):
        subdivide_wake(np.zeros(512 * 6), 512, [0] * 6)
    with pytest.raises(ValueError, match="above 50"):
        emg_envelope(np.ones(200), 20)


def test_noninteger_sampling_rate_keeps_timing():
    fs = 244.140625
    times = np.arange(round(fs * 8)) / fs
    result = subdivide_wake(
        np.sin(2 * np.pi * 60 * times),
        fs,
        [0] * 8,
        config=WakeActivityConfig(method="nrem_envelope", threshold=0.1),
    )
    assert result.sleep_scores.shape == (8,)
    assert np.all(result.sleep_scores == 4)


def test_per_second_rms_rank_targets_eighty_percent_active():
    values = np.arange(1.0, 11.0)
    with patch("app_src.wake_activity.emg_second_rms", return_value=values):
        result = subdivide_wake([], 512, [0] * 10)
    np.testing.assert_array_equal(result.sleep_scores, [5, 5] + [4] * 8)
    assert result.metadata["active_seconds"] == 8
    assert result.metadata["achieved_active_fraction"] == pytest.approx(0.8)
    assert not result.metadata["target_constrained_by_manual_labels"]


def test_per_second_rms_rank_preserves_manual_labels_and_reports_constraint():
    values = np.arange(1.0, 6.0)
    with patch("app_src.wake_activity.emg_second_rms", return_value=values):
        result = subdivide_wake([], 512, [0] * 5, [5, 5, None, None, None])
    np.testing.assert_array_equal(result.sleep_scores, [5, 5, 4, 4, 4])
    assert result.metadata["active_seconds"] == 3
    assert result.metadata["target_constrained_by_manual_labels"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": -1},
        {"threshold": np.inf},
        {"min_duration": 0},
        {"min_duration": np.nan},
        {"nrem_baseline_percentile": 101},
        {"nrem_deviation_multiplier": -1},
        {"method": "unknown"},
        {"active_fraction": 1},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WakeActivityConfig(**kwargs)
