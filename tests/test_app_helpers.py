"""Tests for app helpers in app_src.session, app_src.resampling, and app_src.callbacks."""

from unittest.mock import MagicMock, patch

import dash
import numpy as np


class TestWriteMetadata:
    """Tests for write_metadata function."""

    def test_basic_metadata(self, mock_mat_data):
        """Test basic metadata extraction from mat data."""
        from app_src.session import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "video_start_time" in metadata
        assert "video_path" in metadata

    def test_calculates_duration(self, mock_mat_data):
        """Test that end_time is calculated from EEG duration."""
        from app_src.session import write_metadata

        metadata = write_metadata(mock_mat_data)

        # 100 seconds of EEG at 512 Hz
        expected_duration = 100
        assert metadata["end_time"] == expected_duration

    def test_default_start_time(self, mock_mat_data):
        """Test that start_time defaults to 0."""
        from app_src.session import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["start_time"] == 0

    def test_custom_start_time(self, mock_mat_data):
        """Test with custom start_time in mat data."""
        from app_src.session import write_metadata

        mock_mat_data["start_time"] = 3600  # 1 hour offset
        metadata = write_metadata(mock_mat_data)

        assert metadata["start_time"] == 3600
        assert metadata["end_time"] == 3600 + 100  # start + duration

    def test_video_start_time_default(self, mock_mat_data):
        """Test that video_start_time defaults to 0."""
        from app_src.session import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["video_start_time"] == 0

    def test_video_start_time_accepts_negative_float(self, mock_mat_data):
        """Test that video_start_time preserves signed fractional offsets."""
        from app_src.session import write_metadata

        mock_mat_data["video_start_time"] = -2.5
        metadata = write_metadata(mock_mat_data)

        assert metadata["video_start_time"] == -2.5

    def test_video_path_default(self, mock_mat_data):
        """Test that video_path defaults to empty string."""
        from app_src.session import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["video_path"] == ""


class TestClearTempDir:
    """Tests for clear_temp_dir function."""

    def test_clears_mat_and_xlsx(self, tmp_path):
        """Test that .mat and .xlsx files are cleared except current file."""
        from app_src.session import clear_temp_dir

        # Create temp files
        (tmp_path / "old_file.mat").touch()
        (tmp_path / "old_file.xlsx").touch()
        (tmp_path / "current_file.mat").touch()
        (tmp_path / "keep_this.txt").touch()

        # Patch TEMP_PATH
        with patch("app_src.session.TEMP_PATH", tmp_path):
            clear_temp_dir("current_file")

        # old files should be deleted, current and txt should remain
        assert not (tmp_path / "old_file.mat").exists()
        assert not (tmp_path / "old_file.xlsx").exists()
        assert (tmp_path / "current_file.mat").exists()
        assert (tmp_path / "keep_this.txt").exists()


class TestInitializeCache:
    """Tests for initialize_cache function."""

    def test_sets_filepath(self):
        """Test that filepath is set in cache."""
        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/file.mat")

        # Check filepath was set
        mock_cache.set.assert_any_call("filepath", "/path/to/file.mat")

    def test_sets_filename(self):
        """Test that filename (without extension) is set in cache."""
        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/my_recording.mat")

        # Check filename was set (stem only)
        mock_cache.set.assert_any_call("filename", "my_recording")

    def test_initializes_history_for_new_file(self):
        """Test that sleep_scores_history is reset for new file."""

        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.side_effect = lambda key: {
            "filepath": "/path/to/old_file.mat",
            "recent_files_with_video": None,
            "file_video_record": None,
        }.get(key)

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/new_file.mat")

        # Check history was reset
        calls = [str(c) for c in mock_cache.set.call_args_list]
        assert any("sleep_scores_history" in c for c in calls)

    def test_preserves_history_for_same_file(self):
        """Test that history is preserved when reopening same file."""
        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.side_effect = lambda key: {
            "filepath": "/path/to/./same_file.mat",
            "recent_files_with_video": [],
            "file_video_record": {},
        }.get(key)

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/same_file.mat")

        # History should not be reset (check it wasn't called with deque)
        set_calls = mock_cache.set.call_args_list
        history_calls = [c for c in set_calls if c[0][0] == "sleep_scores_history"]
        # When same file, history is NOT reset
        assert len(history_calls) == 0

    def test_resets_history_for_same_basename_in_different_folder(self):
        """Files with the same basename must not share recovery history."""
        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.side_effect = lambda key: {
            "filepath": "/first/session/recording.mat",
            "recent_files_with_video": [],
            "file_video_record": {},
        }.get(key)

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/second/session/recording.mat")

        history_calls = [
            call for call in mock_cache.set.call_args_list if call.args[0] == "sleep_scores_history"
        ]
        assert len(history_calls) == 1

    def test_older_cache_without_filepath_resets_history(self):
        """A filename-only cache cannot safely identify the recording."""
        from app_src.session import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.side_effect = lambda key: {
            "filename": "recording",
            "recent_files_with_video": [],
            "file_video_record": {},
        }.get(key)

        with patch("app_src.session.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/recording.mat")

        history_calls = [
            call for call in mock_cache.set.call_args_list if call.args[0] == "sleep_scores_history"
        ]
        assert len(history_calls) == 1


class TestSaveAnnotations:
    """Tests for save annotation behavior."""

    def test_cancelled_mat_save_still_reports_unscored_segment(self, tmp_path):
        from app_src.callbacks.saving import save_annotations

        sleep_scores = np.array([0, 1, np.nan, np.nan, 2], dtype=float)
        mat = {"sleep_scores": np.array([0, 1, 2]), "ne": np.array([])}
        cache_values = {
            "filepath": str(tmp_path / "recording.mat"),
            "filename": "recording",
            "sleep_scores_history": [sleep_scores],
        }

        with (
            patch("app_src.callbacks.saving.TEMP_PATH", tmp_path),
            patch(
                "app_src.callbacks.saving.cache.get",
                side_effect=lambda key: cache_values.get(key),
            ),
            patch("app_src.callbacks.saving.loadmat", return_value=mat),
            patch("app_src.callbacks.saving.save_file_dialog", return_value=None),
        ):
            message, max_intervals = save_annotations(1)

        assert (
            message == "Unscored segment found: [2, 4] (2 s). "
            "Complete scoring to export the sleep bout spreadsheet."
        )
        assert max_intervals == 5


class TestMakeClip:
    """Tests for video clip timing guards."""

    def test_rejects_negative_adjusted_start(self, tmp_path):
        from app_src.callbacks.video import make_clip

        with (
            patch("app_src.callbacks.video.VIDEO_DIR", tmp_path),
            patch("app_src.callbacks.video.get_video_duration", return_value=100),
            patch("app_src.callbacks.video.make_mp4_clip") as mock_make_mp4_clip,
        ):
            clip_name, message = make_clip(
                "recording.avi",
                [0, 10],
                {"video_start_time": -5},
            )

        assert clip_name is None
        assert message.startswith("Video clip unavailable:")
        assert "before the video begins" in message
        mock_make_mp4_clip.assert_not_called()

    def test_rejects_adjusted_end_after_video_duration(self, tmp_path):
        from app_src.callbacks.video import make_clip

        with (
            patch("app_src.callbacks.video.VIDEO_DIR", tmp_path),
            patch("app_src.callbacks.video.get_video_duration", return_value=100),
            patch("app_src.callbacks.video.make_mp4_clip") as mock_make_mp4_clip,
        ):
            clip_name, message = make_clip(
                "recording.avi",
                [95, 105],
                {"video_start_time": 0},
            )

        assert clip_name is None
        assert message.startswith("Video clip unavailable:")
        assert "after the video ends" in message
        mock_make_mp4_clip.assert_not_called()

    def test_allows_valid_negative_float_offset(self, tmp_path):
        from app_src.callbacks.video import make_clip

        with (
            patch("app_src.callbacks.video.VIDEO_DIR", tmp_path),
            patch("app_src.callbacks.video.get_video_duration", return_value=100),
            patch("app_src.callbacks.video.make_mp4_clip") as mock_make_mp4_clip,
        ):
            clip_name, message = make_clip(
                "recording.avi",
                [10, 20],
                {"video_start_time": -5.5},
            )

        assert clip_name == "recording_time_range_4.5-14.5.mp4"
        assert message == ""
        mock_make_mp4_clip.assert_called_once()
        assert mock_make_mp4_clip.call_args.kwargs["start_time"] == 4.5
        assert mock_make_mp4_clip.call_args.kwargs["end_time"] == 14.5

    def test_empty_clip_name_clears_video_without_overwriting_message(self):
        from app_src.callbacks.video import show_clip

        title, container, message = show_clip(None)

        assert title == ""
        assert container == ""
        assert message is dash.no_update


class TestDirectRestylePayload:
    """Tests for browser-side Plotly.restyle payload helpers."""

    def test_build_direct_restyle_payload_serializes_patch_operations(self):
        from dash import Patch

        from app_src.resampling import build_direct_restyle_payload

        profile_marker = {"profileId": 12, "mode": "final", "source": "keyboard"}
        patch = Patch()
        patch["data"][0]["x"] = [1.0, 2.0]
        patch["data"][0]["y"] = [3.0, 4.0]
        patch["layout"]["meta"]["sleepScoringNavigationProfile"] = profile_marker

        payload = build_direct_restyle_payload(patch, profile_marker)

        assert payload["applyPath"] == "direct-restyle"
        assert payload["profileMarker"] == profile_marker
        assert payload["operations"][0]["location"] == ["data", 0, "x"]
        assert payload["operations"][1]["location"] == ["data", 0, "y"]
        assert payload["operations"][2]["location"] == [
            "layout",
            "meta",
            "sleepScoringNavigationProfile",
        ]
