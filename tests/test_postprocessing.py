"""Tests for app_src/postprocessing.py"""

import numpy as np
import pandas as pd

from app_src.postprocessing import (
    check_REM_duration,
    edit_sleep_scores,
    get_pred_label_stats,
    get_sleep_segments,
    merge_consecutive_sleep_scores,
    modify_SWS,
)


class TestGetSleepSegments:
    """Tests for get_sleep_segments function."""

    def test_basic_segmentation(self, sample_sleep_scores):
        """Test basic sleep score segmentation."""
        df = get_sleep_segments(sample_sleep_scores)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["sleep_scores", "start", "end", "duration"]
        # Should have 5 segments: Wake, SWS, REM, SWS, Wake
        assert len(df) == 5

    def test_segment_values(self, sample_sleep_scores):
        """Test that segment values are correct."""
        df = get_sleep_segments(sample_sleep_scores)

        # First segment should be Wake (0), start=0
        assert df.iloc[0]["sleep_scores"] == 0
        assert df.iloc[0]["start"] == 0

        # Check duration calculation
        for _, row in df.iterrows():
            assert row["duration"] == row["end"] - row["start"] + 1

    def test_longer_sequence(self, longer_sleep_scores):
        """Test with longer sleep sequence."""
        df = get_sleep_segments(longer_sleep_scores)

        # Should have 5 segments based on fixture pattern
        assert len(df) == 5

        # Total duration should match input length
        total_duration = df["duration"].sum()
        assert total_duration == len(longer_sleep_scores)

    def test_single_stage(self):
        """Test with only one sleep stage."""
        scores = np.ones(10, dtype=int)  # All SWS
        df = get_sleep_segments(scores)

        assert len(df) == 1
        assert df.iloc[0]["sleep_scores"] == 1
        assert df.iloc[0]["duration"] == 10


class TestMergeConsecutiveSleepScores:
    """Tests for merge_consecutive_sleep_scores function."""

    def test_merge_consecutive(self):
        """Test merging consecutive segments of same stage."""
        # Create DataFrame with consecutive same-stage segments
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 0, 1, 1],
                "start": [0, 10, 20, 30],
                "end": [9, 19, 29, 39],
                "duration": [10, 10, 10, 10],
            }
        )

        merged = merge_consecutive_sleep_scores(df)

        # Should merge to 2 segments
        assert len(merged) == 2
        assert merged.iloc[0]["duration"] == 20  # Merged Wake
        assert merged.iloc[1]["duration"] == 20  # Merged SWS

    def test_no_merge_needed(self, sample_sleep_segments_df):
        """Test when no merging is needed."""
        merged = merge_consecutive_sleep_scores(sample_sleep_segments_df)

        # All segments are already different stages
        assert len(merged) == len(sample_sleep_segments_df)

    def test_preserves_order(self):
        """Test that merged segments maintain correct order."""
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 1, 1, 2],
                "start": [0, 10, 20, 30],
                "end": [9, 19, 29, 39],
                "duration": [10, 10, 10, 10],
            }
        )

        merged = merge_consecutive_sleep_scores(df)

        # Check order is maintained
        assert merged.iloc[0]["sleep_scores"] == 0
        assert merged.iloc[1]["sleep_scores"] == 1
        assert merged.iloc[2]["sleep_scores"] == 2


class TestModifySWS:
    """Tests for modify_SWS function."""

    def test_eliminate_short_sws_between_wake(self):
        """Short SWS (<=5s) between Wake segments should become Wake."""
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 1, 0],
                "start": [0, 10, 15],
                "end": [9, 14, 25],
                "duration": [10, 5, 11],  # 5s SWS between Wake
            }
        )

        modified = modify_SWS(df)

        # Short SWS should be changed to Wake
        assert modified.iloc[1]["sleep_scores"] == 0

    def test_keep_longer_sws(self):
        """SWS longer than 5s should not be modified."""
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 1, 0],
                "start": [0, 10, 20],
                "end": [9, 19, 30],
                "duration": [10, 10, 11],  # 10s SWS
            }
        )

        modified = modify_SWS(df)

        # SWS should remain
        assert modified.iloc[1]["sleep_scores"] == 1

    def test_keep_sws_not_between_wake(self):
        """Short SWS not between Wake segments should not be modified."""
        df = pd.DataFrame(
            {
                "sleep_scores": [2, 1, 0],  # REM, SWS, Wake
                "start": [0, 10, 15],
                "end": [9, 14, 25],
                "duration": [10, 5, 11],
            }
        )

        modified = modify_SWS(df)

        # SWS should remain (preceded by REM, not Wake)
        assert modified.iloc[1]["sleep_scores"] == 1


class TestCheckREMDuration:
    """Tests for check_REM_duration function."""

    def test_eliminate_short_rem(self):
        """REM shorter than 7s should be relabeled."""
        df = pd.DataFrame(
            {
                "sleep_scores": [1, 2, 1],
                "start": [0, 10, 16],
                "end": [9, 15, 25],
                "duration": [10, 6, 10],  # 6s REM (too short)
            }
        )

        modified = check_REM_duration(df)

        # Short REM should be changed to neighboring stage (SWS)
        assert modified.iloc[1]["sleep_scores"] == 1

    def test_keep_longer_rem(self):
        """REM 7s or longer should not be modified."""
        df = pd.DataFrame(
            {
                "sleep_scores": [1, 2, 1],
                "start": [0, 10, 17],
                "end": [9, 16, 27],
                "duration": [10, 7, 11],  # 7s REM (ok)
            }
        )

        modified = check_REM_duration(df)

        # REM should remain
        assert modified.iloc[1]["sleep_scores"] == 2


class TestEditSleepScores:
    """Tests for edit_sleep_scores function."""

    def test_apply_changes(self):
        """Test that DataFrame changes are applied to sleep scores array."""
        sleep_scores = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        df = pd.DataFrame(
            {
                "sleep_scores": [1, 1, 2],  # Changed first segment to SWS
                "start": [0, 3, 6],
                "end": [2, 5, 8],
                "duration": [3, 3, 3],
            }
        )

        result = edit_sleep_scores(sleep_scores, df)

        # First 3 should now be SWS (1)
        np.testing.assert_array_equal(result[:3], [1, 1, 1])
        # Original array should be unchanged
        np.testing.assert_array_equal(sleep_scores[:3], [0, 0, 0])

    def test_preserves_original(self):
        """Test that original array is not modified."""
        original = np.array([0, 1, 2])
        df = pd.DataFrame(
            {
                "sleep_scores": [1],
                "start": [0],
                "end": [0],
                "duration": [1],
            }
        )

        _ = edit_sleep_scores(original, df)

        # Original should be unchanged
        assert original[0] == 0


class TestGetPredLabelStats:
    """Tests for get_pred_label_stats function."""

    def test_basic_stats(self, sample_sleep_segments_df):
        """Test basic statistics calculation."""
        stats = get_pred_label_stats(sample_sleep_segments_df.copy())

        assert isinstance(stats, pd.DataFrame)
        assert "Wake" in stats.columns
        assert "SWS" in stats.columns
        assert "REM" in stats.columns
        assert "MA" in stats.columns

    def test_time_percentages_sum_to_100(self, sample_sleep_segments_df):
        """Test that time percentages sum to approximately 100."""
        stats = get_pred_label_stats(sample_sleep_segments_df.copy())

        time_percent_row = stats.loc["Time (%)"]
        total_percent = time_percent_row.sum()
        assert abs(total_percent - 100) < 0.1  # Allow small rounding error

    def test_ma_classification(self):
        """Test that short Wake (<15s) is classified as MA."""
        df = pd.DataFrame(
            {
                "sleep_scores": [0, 1, 0],
                "start": [0, 10, 40],
                "end": [9, 39, 49],
                "duration": [10, 30, 10],  # First and last Wake are <15s -> MA
            }
        )

        stats = get_pred_label_stats(df)

        # Short Wake should be counted as MA
        assert stats.loc["Count", "MA"] == 2
        assert stats.loc["Count", "Wake"] == 0

    def test_stat_index_labels(self, sample_sleep_segments_df):
        """Test that index labels are correct."""
        stats = get_pred_label_stats(sample_sleep_segments_df.copy())

        expected_indices = [
            "Time (s)",
            "Time (%)",
            "Count",
            "Wake Transition Count",
            "SWS Transition Count",
            "REM Transition Count",
            "MA Transition Count",
        ]
        assert list(stats.index) == expected_indices
