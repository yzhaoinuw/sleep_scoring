# -*- coding: utf-8 -*-
"""Tests for the local, aggregate-only usage tracker."""

import json
import multiprocessing
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from app_src import usage_stats


def hold_usage_stats_lock(stats_path, acquired, release):
    """Hold the file lock from a separate process for cross-window coverage."""
    with usage_stats._usage_stats_lock(Path(stats_path)):
        acquired.set()
        release.wait(timeout=5)


def make_mat(eeg_seed=0, scored_seconds=3600, eeg_samples=5000):
    rng = np.random.default_rng(eeg_seed)
    return {
        "eeg": rng.standard_normal(eeg_samples).astype(np.float32),
        "sleep_scores": np.zeros(scored_seconds, dtype=float),
    }


def test_counts_a_fully_scored_recording_once(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    mat = make_mat(scored_seconds=7200)

    assert usage_stats.record_scored_recording(mat, stats_file) is True
    assert usage_stats.record_scored_recording(mat, stats_file) is False

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["recordings_scored"] == 1
    assert stats["seconds_scored"] == 7200


def test_reopening_and_resaving_does_not_inflate_totals(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    mat = make_mat(eeg_seed=7)
    usage_stats.record_scored_recording(mat, stats_file)

    # Saving under a new name, then reopening it, produces an equal but
    # distinct array: identity has to come from the signal, not the object.
    reopened = {
        "eeg": np.array(mat["eeg"], copy=True),
        "sleep_scores": np.array(mat["sleep_scores"], copy=True),
    }
    assert usage_stats.record_scored_recording(reopened, stats_file) is False

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["recordings_scored"] == 1


def test_distinct_recordings_accumulate(tmp_path):
    stats_file = tmp_path / "usage-stats.json"

    usage_stats.record_scored_recording(make_mat(eeg_seed=1, scored_seconds=3600), stats_file)
    usage_stats.record_scored_recording(make_mat(eeg_seed=2, scored_seconds=1800), stats_file)

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["recordings_scored"] == 2
    assert stats["seconds_scored"] == 5400


def test_simultaneous_updates_do_not_lose_a_recording(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    original_write = usage_stats.write_usage_stats
    first_write_started = threading.Event()
    allow_first_write = threading.Event()

    def delayed_write(stats, path):
        if not first_write_started.is_set():
            first_write_started.set()
            assert allow_first_write.wait(timeout=2)
        return original_write(stats, path)

    monkeypatch.setattr(usage_stats, "write_usage_stats", delayed_write)
    results = []

    def record(seed):
        results.append(usage_stats.record_scored_recording(make_mat(eeg_seed=seed), stats_file))

    first = threading.Thread(target=record, args=(1,))
    second = threading.Thread(target=record, args=(2,))
    first.start()
    assert first_write_started.wait(timeout=2)
    second.start()
    time.sleep(0.05)
    allow_first_write.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert results == [True, True]
    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["recordings_scored"] == 2
    assert stats["seconds_scored"] == 7200


def test_process_lock_waits_for_another_app_window(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release = context.Event()
    lock_holder = context.Process(
        target=hold_usage_stats_lock,
        args=(str(stats_file), acquired, release),
    )
    lock_holder.start()
    assert acquired.wait(timeout=5)

    recorded = threading.Event()
    result = []

    def record():
        result.append(usage_stats.record_scored_recording(make_mat(eeg_seed=9), stats_file))
        recorded.set()

    recorder = threading.Thread(target=record)
    recorder.start()
    assert not recorded.wait(timeout=0.1)

    release.set()
    lock_holder.join(timeout=5)
    recorder.join(timeout=5)

    assert lock_holder.exitcode == 0
    assert result == [True]


def test_recording_without_eeg_or_scores_is_not_counted(tmp_path):
    stats_file = tmp_path / "usage-stats.json"

    assert usage_stats.record_scored_recording({"sleep_scores": np.zeros(10)}, stats_file) is False
    assert usage_stats.record_scored_recording({"eeg": np.zeros(10)}, stats_file) is False
    assert usage_stats.record_scored_recording({}, stats_file) is False

    assert not stats_file.exists()


def test_store_holds_no_paths_names_or_signal_values(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    mat = make_mat(eeg_seed=3)
    mat["filename"] = "mouse_42_day7.mat"
    usage_stats.record_scored_recording(mat, stats_file)

    stored_text = stats_file.read_text(encoding="utf-8")
    assert "mouse_42" not in stored_text
    assert ".mat" not in stored_text

    stored = json.loads(stored_text)
    assert set(stored) == {
        "schema_version",
        "recordings_scored",
        "seconds_scored",
        "first_recorded_at",
        "last_recorded_at",
        "counted_recordings",
    }
    # Fingerprints are one-way digests, never the signal itself.
    assert all(
        len(fingerprint) == usage_stats.FINGERPRINT_LENGTH
        for fingerprint in stored["counted_recordings"]
    )


def test_exported_summary_carries_counts_but_no_fingerprints(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    usage_stats.record_scored_recording(make_mat(eeg_seed=4, scored_seconds=5400), stats_file)
    stats = usage_stats.read_usage_stats(stats_file)

    summary = usage_stats.format_usage_summary(stats, app_version="0.17.0")

    assert "Recordings scored:    1" in summary
    assert "Hours scored:         1.5" in summary
    assert "0.17.0" in summary
    for fingerprint in stats["counted_recordings"]:
        assert fingerprint not in summary


def test_summary_reads_cleanly_before_any_recording_is_scored(tmp_path):
    summary = usage_stats.format_usage_summary(
        usage_stats.read_usage_stats(tmp_path / "missing.json")
    )

    assert "Recordings scored:    0" in summary
    assert "Hours scored:         0.0" in summary
    assert "n/a" in summary


@pytest.mark.parametrize("contents", ["{not json", "[]", '{"schema_version": 99}'])
def test_unreadable_or_foreign_store_is_ignored_safely(tmp_path, contents):
    stats_file = tmp_path / "usage-stats.json"
    stats_file.write_text(contents, encoding="utf-8")

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats == usage_stats.get_empty_usage_stats()

    # A damaged store must not block later counting.
    assert usage_stats.record_scored_recording(make_mat(eeg_seed=5), stats_file) is True


def test_counting_never_raises_when_the_store_is_unwritable(tmp_path):
    unwritable = tmp_path / "not-a-directory"
    unwritable.write_text("blocking file", encoding="utf-8")

    assert usage_stats.record_scored_recording(make_mat(), unwritable / "stats.json") is False


def test_stats_file_defaults_to_a_per_user_path_outside_the_app(monkeypatch, tmp_path):
    monkeypatch.delenv(usage_stats.USAGE_STATS_FILE_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert usage_stats.get_usage_stats_file() == tmp_path / "sleep_scoring" / "usage-stats.json"


def test_stats_file_can_be_overridden_for_testing(monkeypatch, tmp_path):
    override = tmp_path / "custom-usage.json"
    monkeypatch.setenv(usage_stats.USAGE_STATS_FILE_ENV, str(override))

    assert usage_stats.get_usage_stats_file() == override
