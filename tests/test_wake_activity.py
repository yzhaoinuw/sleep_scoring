from unittest.mock import patch

import numpy as np
import pytest

from app_src.wake_activity import (
    WakeActivityConfig,
    active_mask,
    emg_envelope,
    second_activity,
    subdivide_wake,
)


def test_duration_boundary_and_interior_gap():
    config = WakeActivityConfig(threshold=2)
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
        result = subdivide_wake([], 512, coarse, users, WakeActivityConfig(threshold=2))
    np.testing.assert_array_equal(result.sleep_scores, [4] * 6 + [1, 2, 3])
    assert result.calibrated_seconds == 0
    np.testing.assert_array_equal(coarse, [0] * 6 + [1, 2, 3])


def test_fine_examples_calibrate_before_override_and_force_coarse_wake():
    users = [5] + [None] * 4 + [4] + [None] * 4
    envelope = np.r_[np.ones(100), np.full(100, 6.0)]
    with patch("app_src.wake_activity.emg_envelope", return_value=envelope):
        result = subdivide_wake(
            [], 512, [1] + [0] * 4 + [2] + [0] * 4, users, WakeActivityConfig(threshold=100)
        )
    assert 1 <= result.threshold < 6  # Cannot pass by merely overlaying the examples.
    assert result.calibrated_seconds == 2
    np.testing.assert_array_equal(result.sleep_scores, [5] * 5 + [4] * 5)


def test_manual_subtype_shorter_than_minimum_remains_authoritative():
    with patch("app_src.wake_activity.emg_envelope", return_value=np.ones(40)):
        result = subdivide_wake([], 512, [0, 0], [4, None])
    np.testing.assert_array_equal(result.sleep_scores, [4, 5])
    assert result.active_bouts == []


def test_missing_emg_fails_in_wake_instead_of_becoming_quiet():
    with patch("app_src.wake_activity.emg_envelope", return_value=np.r_[np.ones(20), np.nan]):
        with pytest.raises(ValueError, match="invalid EMG"):
            subdivide_wake([], 512, [0, 0])


def test_invalid_emg_during_sleep_does_not_block_valid_wake():
    with patch(
        "app_src.wake_activity.emg_envelope", return_value=np.r_[np.ones(120), [np.nan] * 20]
    ):
        result = subdivide_wake([], 512, [0] * 6 + [1], config=WakeActivityConfig(threshold=2))
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


def test_flatline_and_low_sampling_rate_are_not_silent_quiet_predictions():
    with pytest.raises(ValueError, match="flatlined"):
        subdivide_wake(np.zeros(512 * 6), 512, [0] * 6)
    with pytest.raises(ValueError, match="above 50"):
        emg_envelope(np.ones(200), 20)


def test_noninteger_sampling_rate_keeps_timing():
    fs = 244.140625
    times = np.arange(round(fs * 8)) / fs
    result = subdivide_wake(
        np.sin(2 * np.pi * 60 * times), fs, [0] * 8, config=WakeActivityConfig(threshold=0.1)
    )
    assert result.sleep_scores.shape == (8,)
    assert np.all(result.sleep_scores == 4)


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": -1}, {"threshold": np.inf}, {"min_duration": 0}, {"min_duration": np.nan}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WakeActivityConfig(**kwargs)
