# -*- coding: utf-8 -*-
"""What the save callback actually counts as a scored recording."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app_src import usage_stats


@pytest.fixture
def saved_recording(tmp_path, monkeypatch):
    """Drive save_annotations against fakes and report what got counted."""
    from app_src.callbacks import saving

    stats_file = tmp_path / "usage-stats.json"
    monkeypatch.setenv(usage_stats.USAGE_STATS_FILE_ENV, str(stats_file))

    def run(sleep_scores, save_accepted=True):
        rng = np.random.default_rng(0)
        mat = {
            "eeg": rng.standard_normal(4000).astype(np.float32),
            "eeg_frequency": 512.0,
            "sleep_scores": np.asarray(sleep_scores, dtype=float),
        }
        cache = MagicMock()
        cache.get.side_effect = lambda key: {
            "filepath": str(tmp_path / "recording.mat"),
            "filename": "recording",
            "sleep_scores_history": [np.asarray(sleep_scores, dtype=float)],
        }[key]

        with (
            patch.object(saving, "cache", cache),
            patch.object(saving, "TEMP_PATH", tmp_path),
            patch.object(saving, "loadmat", return_value=mat),
            patch.object(saving, "savemat"),
            patch.object(saving, "shutil"),
            patch.object(
                saving,
                "save_file_dialog",
                return_value=str(tmp_path / "out.mat") if save_accepted else None,
            ),
        ):
            saving.save_annotations(1)

        return usage_stats.read_usage_stats(stats_file)

    return run


def test_fully_scored_and_saved_recording_is_counted(saved_recording):
    stats = saved_recording(np.zeros(3600))

    assert stats["recordings_scored"] == 1
    assert stats["seconds_scored"] == 3600


def test_partially_scored_recording_is_not_counted(saved_recording):
    # -1 marks an unscored second, so this recording is not finished.
    sleep_scores = np.zeros(3600)
    sleep_scores[1000:1200] = -1

    stats = saved_recording(sleep_scores)

    assert stats["recordings_scored"] == 0
    assert stats["seconds_scored"] == 0


def test_cancelled_save_is_not_counted(saved_recording):
    stats = saved_recording(np.zeros(3600), save_accepted=False)

    assert stats["recordings_scored"] == 0
