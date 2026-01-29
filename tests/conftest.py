"""Shared fixtures for sleep scoring tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_sleep_scores():
    """Simple sleep scores array for testing.

    Pattern: Wake(0), Wake(0), SWS(1), SWS(1), SWS(1), REM(2), REM(2), SWS(1), Wake(0)
    """
    return np.array([0, 0, 1, 1, 1, 2, 2, 1, 0])


@pytest.fixture
def longer_sleep_scores():
    """Longer sleep scores for more realistic testing.

    100 seconds of data with varied sleep stages.
    """
    # Create pattern: 20s Wake, 30s SWS, 20s REM, 15s SWS, 15s Wake
    scores = np.concatenate(
        [
            np.zeros(20, dtype=int),  # Wake
            np.ones(30, dtype=int),  # SWS
            np.full(20, 2, dtype=int),  # REM
            np.ones(15, dtype=int),  # SWS
            np.zeros(15, dtype=int),  # Wake
        ]
    )
    return scores


@pytest.fixture
def sleep_scores_with_missing():
    """Sleep scores with missing labels (-1 and nan) at edges."""
    return np.array([-1, -1, 0, 1, 1, 2, 0, -1, np.nan])


@pytest.fixture
def mock_eeg_signal():
    """Mock EEG signal (100 seconds at 512 Hz)."""
    np.random.seed(42)
    return np.random.randn(512 * 100).astype(np.float32)


@pytest.fixture
def mock_emg_signal():
    """Mock EMG signal (100 seconds at 512 Hz)."""
    np.random.seed(43)
    return np.random.randn(512 * 100).astype(np.float32)


@pytest.fixture
def mock_ne_signal():
    """Mock NE signal (100 seconds at 10 Hz)."""
    np.random.seed(44)
    return np.random.randn(10 * 100).astype(np.float32)


@pytest.fixture
def mock_mat_data(mock_eeg_signal, mock_emg_signal, longer_sleep_scores):
    """Minimal mat file structure for testing (no NE)."""
    return {
        "eeg": mock_eeg_signal,
        "emg": mock_emg_signal,
        "eeg_frequency": 512,
        "sleep_scores": longer_sleep_scores,
    }


@pytest.fixture
def mock_mat_data_with_ne(mock_mat_data, mock_ne_signal):
    """Mat file structure with NE signal."""
    return {
        **mock_mat_data,
        "ne": mock_ne_signal,
        "ne_frequency": 10,
    }


@pytest.fixture
def sample_sleep_segments_df():
    """Pre-computed sleep segments DataFrame for testing."""
    return pd.DataFrame(
        {
            "sleep_scores": [0, 1, 2, 1, 0],
            "start": [0, 20, 50, 70, 85],
            "end": [19, 49, 69, 84, 99],
            "duration": [20, 30, 20, 15, 15],
        }
    )
