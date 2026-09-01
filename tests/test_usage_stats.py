# -*- coding: utf-8 -*-
"""Tests for the local, aggregate-only usage tracker."""

import json
import multiprocessing
import threading
import time
from pathlib import Path
from urllib.error import HTTPError

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
        "reporting_enabled",
        "app_instance_id",
        "pending_reports",
        "deferred_recordings_delta",
        "deferred_seconds_delta",
    }
    # Fingerprints are one-way digests, never the signal itself.
    assert all(
        len(fingerprint) == usage_stats.FINGERPRINT_LENGTH
        for fingerprint in stored["counted_recordings"]
    )


@pytest.mark.parametrize("contents", ["{not json", "[]", '{"schema_version": 99}'])
def test_unreadable_or_foreign_store_is_ignored_safely(tmp_path, contents):
    stats_file = tmp_path / "usage-stats.json"
    stats_file.write_text(contents, encoding="utf-8")

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats == usage_stats.get_empty_usage_stats()

    # A damaged store must not block later counting.
    assert usage_stats.record_scored_recording(make_mat(eeg_seed=5), stats_file) is True


def test_transient_store_read_error_does_not_overwrite_existing_totals(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    assert usage_stats.record_scored_recording(make_mat(eeg_seed=5), stats_file)
    before = stats_file.read_text(encoding="utf-8")

    def fail_read(*_args, **_kwargs):
        raise PermissionError("temporarily locked")

    monkeypatch.setattr(Path, "read_text", fail_read)
    assert usage_stats.record_scored_recording(make_mat(eeg_seed=6), stats_file) is False
    monkeypatch.undo()

    assert stats_file.read_text(encoding="utf-8") == before


def test_schema_one_store_upgrades_without_losing_totals(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    stats_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recordings_scored": 5,
                "seconds_scored": 18_000,
                "counted_recordings": ["abc"],
            }
        ),
        encoding="utf-8",
    )

    stats = usage_stats.read_usage_stats(stats_file)

    assert stats["schema_version"] == usage_stats.SCHEMA_VERSION
    assert stats["recordings_scored"] == 5
    assert stats["seconds_scored"] == 18_000
    assert stats["counted_recordings"] == ["abc"]
    assert stats["deferred_recordings_delta"] == 0


def test_counting_never_raises_when_the_store_is_unwritable(tmp_path):
    unwritable = tmp_path / "not-a-directory"
    unwritable.write_text("blocking file", encoding="utf-8")

    assert usage_stats.record_scored_recording(make_mat(), unwritable / "stats.json") is False


def test_stats_file_defaults_to_the_shared_app_folder(monkeypatch, tmp_path):
    monkeypatch.delenv(usage_stats.USAGE_STATS_FILE_ENV, raising=False)
    monkeypatch.setenv(usage_stats.APP_STATE_DIR_ENV, str(tmp_path))

    assert usage_stats.get_usage_stats_file() == tmp_path / "usage-stats.json"


def test_stats_file_can_be_overridden_for_testing(monkeypatch, tmp_path):
    override = tmp_path / "custom-usage.json"
    monkeypatch.setenv(usage_stats.USAGE_STATS_FILE_ENV, str(override))

    assert usage_stats.get_usage_stats_file() == override


class SuccessfulResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return 201


def test_config_opted_in_app_queues_anonymous_aggregate_reports(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    first_mat = make_mat(eeg_seed=10, scored_seconds=3600)
    second_mat = make_mat(eeg_seed=11, scored_seconds=1800)
    assert usage_stats.record_scored_recording(first_mat, stats_file)

    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )
    assert usage_stats.configure_usage_reporting(stats_file)
    assert usage_stats.record_scored_recording(second_mat, stats_file)

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["reporting_enabled"] is True
    assert len(stats["app_instance_id"]) == 36
    assert [report["event_kind"] for report in stats["pending_reports"]] == [
        "enrollment",
        "recording",
    ]
    assert stats["pending_reports"][0]["recordings_delta"] == 1
    assert stats["pending_reports"][1]["seconds_delta"] == 1800
    serialized_reports = json.dumps(stats["pending_reports"])
    assert "mouse_42" not in serialized_reports
    for fingerprint in stats["counted_recordings"]:
        assert fingerprint not in serialized_reports


def test_disabled_config_never_activates_reporting(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", False)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )

    assert usage_stats.configure_usage_reporting(stats_file) is False
    assert not stats_file.exists()


def test_configured_sync_sends_each_queued_event_once_and_then_clears_it(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    usage_stats.record_scored_recording(make_mat(eeg_seed=12), stats_file)
    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )
    usage_stats.configure_usage_reporting(stats_file)
    sent_payloads = []

    def opener(request, timeout):
        assert timeout == usage_stats.USAGE_REPORT_TIMEOUT_SECONDS
        sent_payloads.append(json.loads(request.data.decode("utf-8")))
        return SuccessfulResponse()

    assert (
        usage_stats.sync_usage_reports(
            stats_file,
            report_url="https://usage.example/v1/usage-events",
            opener=opener,
        )
        == "sent"
    )
    assert usage_stats.read_usage_stats(stats_file)["pending_reports"] == []
    assert len(sent_payloads) == 1
    assert set(sent_payloads[0]) == {
        "event_id",
        "app_instance_id",
        "event_kind",
        "recordings_delta",
        "seconds_delta",
        "occurred_at",
        "app_version",
    }


def test_permanent_client_error_does_not_block_later_reports(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    usage_stats.record_scored_recording(make_mat(eeg_seed=14), stats_file)
    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )
    assert usage_stats.configure_usage_reporting(stats_file)
    assert usage_stats.record_scored_recording(make_mat(eeg_seed=15), stats_file)
    sent_payloads = []

    def opener(request, timeout):
        assert timeout == usage_stats.USAGE_REPORT_TIMEOUT_SECONDS
        if not sent_payloads:
            sent_payloads.append("rejected")
            raise HTTPError(request.full_url, 400, "bad request", None, None)
        sent_payloads.append(json.loads(request.data.decode("utf-8")))
        return SuccessfulResponse()

    assert usage_stats.sync_usage_reports(stats_file, opener=opener) == "sent"
    assert usage_stats.read_usage_stats(stats_file)["pending_reports"] == []
    assert sent_payloads[1]["event_kind"] == "recording"


def test_pending_reports_are_bounded_and_deferred_totals_are_preserved(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    monkeypatch.setattr(usage_stats, "MAX_PENDING_REPORTS", 1)
    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )
    assert usage_stats.configure_usage_reporting(stats_file)

    for seed in range(20, 23):
        assert usage_stats.record_scored_recording(make_mat(eeg_seed=seed), stats_file)

    stats = usage_stats.read_usage_stats(stats_file)
    assert len(stats["pending_reports"]) == 1
    assert stats["deferred_recordings_delta"] == 2
    assert stats["deferred_seconds_delta"] == 7200

    assert (
        usage_stats.sync_usage_reports(
            stats_file, opener=lambda _request, timeout: SuccessfulResponse()
        )
        == "sent"
    )
    stats = usage_stats.read_usage_stats(stats_file)
    assert len(stats["pending_reports"]) == 1
    assert stats["pending_reports"][0]["recordings_delta"] == 2
    assert stats["deferred_recordings_delta"] == 0


def test_disabled_config_mirrors_state_without_dropping_pending_reports(tmp_path, monkeypatch):
    stats_file = tmp_path / "usage-stats.json"
    usage_stats.record_scored_recording(make_mat(eeg_seed=16), stats_file)
    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    monkeypatch.setattr(
        usage_stats.config, "USAGE_REPORT_URL", "https://usage.example/v1/usage-events"
    )
    assert usage_stats.configure_usage_reporting(stats_file)
    pending_reports = list(usage_stats.read_usage_stats(stats_file)["pending_reports"])

    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", False)
    assert usage_stats.configure_usage_reporting(stats_file) is False

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["reporting_enabled"] is False
    assert stats["pending_reports"] == pending_reports

    monkeypatch.setattr(usage_stats.config, "ENABLE_USAGE_REPORTING", True)
    assert usage_stats.configure_usage_reporting(stats_file) is True
    assert usage_stats.read_usage_stats(stats_file)["pending_reports"] == pending_reports


def test_disabled_reporting_never_creates_uploads(tmp_path):
    stats_file = tmp_path / "usage-stats.json"
    usage_stats.record_scored_recording(make_mat(eeg_seed=13), stats_file)

    stats = usage_stats.read_usage_stats(stats_file)
    assert stats["reporting_enabled"] is False
    assert stats["app_instance_id"] == ""
    assert stats["pending_reports"] == []
