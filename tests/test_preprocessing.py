"""Tests for app_src/preprocessing.py"""

import numpy as np

from app_src.preprocessing import reshape_sleep_data, reshape_sleep_data_ne, trim_missing_labels


class TestTrimMissingLabels:
    """Tests for trim_missing_labels function."""

    def test_trim_back_only(self):
        """Trim trailing -1 values (default behavior)."""
        arr = np.array([0, 1, 1, 2, -1, -1])
        result = trim_missing_labels(arr, trim="b")
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 2]))

    def test_trim_front_only(self):
        """Trim leading -1 values."""
        arr = np.array([-1, -1, 0, 1, 1, 2])
        result = trim_missing_labels(arr, trim="f")
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 2]))

    def test_trim_both(self):
        """Trim both leading and trailing -1 values."""
        arr = np.array([-1, -1, 0, 1, 1, 2, -1])
        result = trim_missing_labels(arr, trim="fb")
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 2]))

    def test_trim_nan_values(self):
        """Trim trailing nan values."""
        arr = np.array([0, 1, 1, 2, np.nan, np.nan])
        result = trim_missing_labels(arr, trim="b")
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 2]))

    def test_trim_mixed_missing(self):
        """Trim mixed -1 and nan values."""
        arr = np.array([-1, np.nan, 0, 1, 2, -1, np.nan])
        result = trim_missing_labels(arr, trim="fb")
        np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    def test_no_trimming_needed(self):
        """Array with no missing values at edges."""
        arr = np.array([0, 1, 1, 2])
        result = trim_missing_labels(arr, trim="fb")
        np.testing.assert_array_equal(result, np.array([0, 1, 1, 2]))

    def test_case_insensitive(self):
        """Trim parameter should be case insensitive."""
        arr = np.array([-1, 0, 1, -1])
        result_lower = trim_missing_labels(arr, trim="fb")
        result_upper = trim_missing_labels(arr, trim="FB")
        np.testing.assert_array_equal(result_lower, result_upper)


class TestReshapeSleepData:
    """Tests for reshape_sleep_data function."""

    def test_reshape_basic(self, mock_mat_data):
        """Test basic reshaping without standardization."""
        eeg, emg, scores = reshape_sleep_data(mock_mat_data, segment_size=512)

        # Should have 100 seconds worth of segments
        assert eeg.shape[0] == 100
        assert eeg.shape[1] == 512
        assert emg.shape[0] == 100
        assert emg.shape[1] == 512

    def test_reshape_without_labels(self, mock_mat_data):
        """Test reshaping without returning labels."""
        result = reshape_sleep_data(mock_mat_data, segment_size=512, has_labels=False)

        # Should return only eeg and emg
        assert len(result) == 2
        eeg, emg = result
        assert eeg.shape[0] == 100
        assert emg.shape[0] == 100

    def test_reshape_with_standardization(self, mock_mat_data):
        """Test that standardization normalizes the signals."""
        eeg, emg, _ = reshape_sleep_data(mock_mat_data, segment_size=512, standardize=True)

        # Standardized data should have mean ~0 and std ~1
        # (approximately, due to reshaping)
        assert abs(np.mean(eeg)) < 0.5
        assert abs(np.mean(emg)) < 0.5


class TestReshapeSleepDataNE:
    """Tests for reshape_sleep_data_ne function."""

    def test_reshape_with_ne(self, mock_mat_data_with_ne):
        """Test reshaping with NE signal."""
        eeg, emg, ne, scores = reshape_sleep_data_ne(
            mock_mat_data_with_ne, segment_size=512, segment_size_ne=10
        )

        # Should have 100 seconds worth of segments
        assert eeg.shape[0] == 100
        assert eeg.shape[1] == 512
        assert emg.shape[0] == 100
        assert ne.shape[0] == 100
        assert ne.shape[1] == 10  # NE at 10 Hz

    def test_reshape_ne_without_labels(self, mock_mat_data_with_ne):
        """Test reshaping without returning labels."""
        result = reshape_sleep_data_ne(mock_mat_data_with_ne, has_labels=False)

        assert len(result) == 3
        eeg, emg, ne = result
        assert eeg.shape[0] == 100
        assert ne.shape[0] == 100

    def test_reshape_without_ne_signal(self, mock_mat_data):
        """Test that missing NE signal creates zero array."""
        # mock_mat_data doesn't have NE
        eeg, emg, ne, scores = reshape_sleep_data_ne(mock_mat_data)

        # NE should be zeros when not present
        assert ne.shape[0] == 100
        assert ne.shape[1] == 10
        np.testing.assert_array_equal(ne, np.zeros_like(ne))
