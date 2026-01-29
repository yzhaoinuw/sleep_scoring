"""Tests for helper functions in app_src/app_dev.py"""

from unittest.mock import MagicMock, patch


class TestWriteMetadata:
    """Tests for write_metadata function."""

    def test_basic_metadata(self, mock_mat_data):
        """Test basic metadata extraction from mat data."""
        from app_src.app_dev import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert "start_time" in metadata
        assert "end_time" in metadata
        assert "video_start_time" in metadata
        assert "video_path" in metadata

    def test_calculates_duration(self, mock_mat_data):
        """Test that end_time is calculated from EEG duration."""
        from app_src.app_dev import write_metadata

        metadata = write_metadata(mock_mat_data)

        # 100 seconds of EEG at 512 Hz
        expected_duration = 100
        assert metadata["end_time"] == expected_duration

    def test_default_start_time(self, mock_mat_data):
        """Test that start_time defaults to 0."""
        from app_src.app_dev import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["start_time"] == 0

    def test_custom_start_time(self, mock_mat_data):
        """Test with custom start_time in mat data."""
        from app_src.app_dev import write_metadata

        mock_mat_data["start_time"] = 3600  # 1 hour offset
        metadata = write_metadata(mock_mat_data)

        assert metadata["start_time"] == 3600
        assert metadata["end_time"] == 3600 + 100  # start + duration

    def test_video_start_time_default(self, mock_mat_data):
        """Test that video_start_time defaults to 0."""
        from app_src.app_dev import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["video_start_time"] == 0

    def test_video_path_default(self, mock_mat_data):
        """Test that video_path defaults to empty string."""
        from app_src.app_dev import write_metadata

        metadata = write_metadata(mock_mat_data)

        assert metadata["video_path"] == ""


class TestClearTempDir:
    """Tests for clear_temp_dir function."""

    def test_clears_mat_and_xlsx(self, tmp_path):
        """Test that .mat and .xlsx files are cleared except current file."""
        from app_src.app_dev import clear_temp_dir

        # Create temp files
        (tmp_path / "old_file.mat").touch()
        (tmp_path / "old_file.xlsx").touch()
        (tmp_path / "current_file.mat").touch()
        (tmp_path / "keep_this.txt").touch()

        # Patch TEMP_PATH
        with patch("app_src.app_dev.TEMP_PATH", tmp_path):
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
        from app_src.app_dev import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("app_src.app_dev.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/file.mat")

        # Check filepath was set
        mock_cache.set.assert_any_call("filepath", "/path/to/file.mat")

    def test_sets_filename(self):
        """Test that filename (without extension) is set in cache."""
        from app_src.app_dev import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch("app_src.app_dev.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/my_recording.mat")

        # Check filename was set (stem only)
        mock_cache.set.assert_any_call("filename", "my_recording")

    def test_initializes_history_for_new_file(self):
        """Test that sleep_scores_history is reset for new file."""

        from app_src.app_dev import initialize_cache

        mock_cache = MagicMock()
        mock_cache.get.side_effect = lambda key: {
            "filename": "old_file",  # Different from new file
            "recent_files_with_video": None,
            "file_video_record": None,
        }.get(key)

        with patch("app_src.app_dev.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/new_file.mat")

        # Check history was reset
        calls = [str(c) for c in mock_cache.set.call_args_list]
        assert any("sleep_scores_history" in c for c in calls)

    def test_preserves_history_for_same_file(self):
        """Test that history is preserved when reopening same file."""
        from app_src.app_dev import initialize_cache

        mock_cache = MagicMock()
        # Same filename as the one being opened
        mock_cache.get.side_effect = lambda key: {
            "filename": "same_file",
            "recent_files_with_video": [],
            "file_video_record": {},
        }.get(key)

        with patch("app_src.app_dev.clear_temp_dir"):
            initialize_cache(mock_cache, "/path/to/same_file.mat")

        # History should not be reset (check it wasn't called with deque)
        set_calls = mock_cache.set.call_args_list
        history_calls = [c for c in set_calls if c[0][0] == "sleep_scores_history"]
        # When same file, history is NOT reset
        assert len(history_calls) == 0
