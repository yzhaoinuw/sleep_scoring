# -*- coding: utf-8 -*-
"""Per-app usage totals and opt-in, aggregate usage-report delivery.

The state belongs beside the app rather than to a Windows account. One shared
app folder, including one on an external drive, therefore has one anonymous
app-instance ID and one set of totals. Reporting is off by default. When it is
enabled, the only network payload is an idempotent aggregate event: an opaque
app ID, an opaque event ID, the number of completed recordings, their scored
seconds, and a timestamp. File names, paths, signal values, annotations,
animal identifiers, and local recording fingerprints never leave the app.
"""

import hashlib
import json
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from app_src import VERSION
from app_src import config


SCHEMA_VERSION = 2
USAGE_STATS_FILE_ENV = "SLEEP_SCORING_USAGE_STATS_FILE"
APP_STATE_DIR_ENV = "SLEEP_SCORING_APP_STATE_DIR"
FINGERPRINT_LENGTH = 16
SECONDS_PER_HOUR = 3600
USAGE_STATS_LOCK_TIMEOUT_SECONDS = 5
USAGE_STATS_LOCK_RETRY_SECONDS = 0.05
USAGE_REPORT_TIMEOUT_SECONDS = 3

try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on macOS/Linux
    import fcntl


_usage_stats_thread_lock = threading.Lock()


def get_app_state_dir():
    """Return the writable directory identifying this copy of the app."""
    configured_dir = os.environ.get(APP_STATE_DIR_ENV)
    if configured_dir:
        return Path(configured_dir)
    return Path(__file__).resolve().parent.parent


def get_usage_stats_file():
    """Return the per-app state file, unless an explicit test override exists."""
    configured_path = os.environ.get(USAGE_STATS_FILE_ENV)
    if configured_path:
        return Path(configured_path)
    return get_app_state_dir() / "usage-stats.json"


def get_usage_report_url():
    """Return the opt-in report endpoint configured in ``config.py``."""
    if not config.ENABLE_USAGE_REPORTING:
        return ""
    return str(config.USAGE_REPORT_URL).strip()


def configure_usage_reporting(stats_file=None):
    """Register this app copy when its ``config.py`` opt-in is enabled."""
    if not get_usage_report_url():
        return False
    return enable_usage_reporting(stats_file)


def get_empty_usage_stats():
    return {
        "schema_version": SCHEMA_VERSION,
        "recordings_scored": 0,
        "seconds_scored": 0,
        "first_recorded_at": "",
        "last_recorded_at": "",
        "counted_recordings": [],
        "reporting_enabled": False,
        "app_instance_id": "",
        "pending_reports": [],
    }


def get_recording_fingerprint(mat):
    """Return a short one-way digest used only for local deduplication."""
    eeg = mat.get("eeg") if hasattr(mat, "get") else None
    if eeg is None:
        return ""

    eeg_array = np.ascontiguousarray(eeg)
    if eeg_array.size == 0:
        return ""

    digest = hashlib.sha256()
    digest.update(str(eeg_array.dtype).encode("utf-8"))
    digest.update(eeg_array.tobytes())
    return digest.hexdigest()[:FINGERPRINT_LENGTH]


def get_scored_seconds(mat):
    """Return the scored duration in seconds, one sleep score per second."""
    sleep_scores = mat.get("sleep_scores") if hasattr(mat, "get") else None
    if sleep_scores is None:
        return 0
    return int(np.asarray(sleep_scores).ravel().size)


def read_usage_stats(stats_file=None):
    """Return usable state, upgrading the original local-only schema in memory."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    stats = get_empty_usage_stats()
    try:
        stored = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stats

    if not isinstance(stored, dict) or stored.get("schema_version") not in (1, SCHEMA_VERSION):
        return stats

    for key, empty_value in stats.items():
        value = stored.get(key)
        if isinstance(value, type(empty_value)):
            stats[key] = value
    stats["schema_version"] = SCHEMA_VERSION
    stats["counted_recordings"] = [str(fingerprint) for fingerprint in stats["counted_recordings"]]
    stats["app_instance_id"] = str(stats["app_instance_id"])
    stats["pending_reports"] = [
        report for report in stats["pending_reports"] if _is_usage_report(report)
    ]
    return stats


def enable_usage_reporting(stats_file=None):
    """Opt this app copy into aggregate reporting and queue its current total."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    try:
        with _usage_stats_lock(stats_path):
            stats = read_usage_stats(stats_path)
            if not stats["app_instance_id"]:
                stats["app_instance_id"] = str(uuid.uuid4())
            if stats["reporting_enabled"]:
                return True

            stats["reporting_enabled"] = True
            if stats["recordings_scored"]:
                stats["pending_reports"].append(
                    _new_usage_report(
                        stats,
                        stats["recordings_scored"],
                        stats["seconds_scored"],
                        event_kind="enrollment",
                    )
                )
            return write_usage_stats(stats, stats_path)
    except Exception:  # noqa: BLE001 -- reporting must never break the app
        return False


def disable_usage_reporting(stats_file=None):
    """Stop future delivery while preserving the app's local totals."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    try:
        with _usage_stats_lock(stats_path):
            stats = read_usage_stats(stats_path)
            stats["reporting_enabled"] = False
            stats["pending_reports"] = []
            return write_usage_stats(stats, stats_path)
    except Exception:  # noqa: BLE001 -- reporting must never break the app
        return False


def record_scored_recording(mat, stats_file=None):
    """Count a fully scored recording once and queue an opted-in report.

    The function never raises: an uncounted recording is preferable to an
    interrupted save. Uploading happens separately, so a slow or unavailable
    network never delays saving annotations.
    """
    fingerprint = get_recording_fingerprint(mat)
    scored_seconds = get_scored_seconds(mat)
    if not fingerprint or scored_seconds <= 0:
        return False

    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    try:
        with _usage_stats_lock(stats_path):
            return _record_scored_recording(fingerprint, scored_seconds, stats_path)
    except Exception:  # noqa: BLE001 -- accounting must never interrupt a save
        return False


def _record_scored_recording(fingerprint, scored_seconds, stats_path):
    stats = read_usage_stats(stats_path)
    if fingerprint in stats["counted_recordings"]:
        return False

    now = _utc_now()
    stats["recordings_scored"] += 1
    stats["seconds_scored"] += scored_seconds
    stats["counted_recordings"].append(fingerprint)
    stats["first_recorded_at"] = stats["first_recorded_at"] or now
    stats["last_recorded_at"] = now
    if stats["reporting_enabled"] and get_usage_report_url():
        if not stats["app_instance_id"]:
            stats["app_instance_id"] = str(uuid.uuid4())
        stats["pending_reports"].append(_new_usage_report(stats, 1, scored_seconds))

    return write_usage_stats(stats, stats_path)


def sync_usage_reports(stats_file=None, report_url=None, opener=urlopen):
    """Send queued, opt-in events once each; return a compact status string."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    endpoint = report_url or get_usage_report_url()
    if not endpoint:
        return "not-configured"

    stats = read_usage_stats(stats_path)
    if not stats["reporting_enabled"]:
        return "disabled"
    reports = list(stats["pending_reports"])
    if not reports:
        return "up-to-date"

    sent_count = 0
    for report in reports:
        if not _send_usage_report(endpoint, report, opener):
            break
        if _remove_pending_report(report["event_id"], stats_path):
            sent_count += 1
    return "sent" if sent_count == len(reports) else "pending"


def _new_usage_report(stats, recordings_delta, seconds_delta, event_kind="recording"):
    return {
        "event_id": str(uuid.uuid4()),
        "app_instance_id": stats["app_instance_id"],
        "event_kind": event_kind,
        "recordings_delta": int(recordings_delta),
        "seconds_delta": int(seconds_delta),
        "occurred_at": _utc_now(),
        "app_version": VERSION,
    }


def _is_usage_report(report):
    return isinstance(report, dict) and {
        "event_id",
        "app_instance_id",
        "event_kind",
        "recordings_delta",
        "seconds_delta",
        "occurred_at",
        "app_version",
    } <= set(report)


def _send_usage_report(endpoint, report, opener):
    payload = json.dumps(report).encode("utf-8")
    request = Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"Sleep-Scoring/{VERSION}",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=USAGE_REPORT_TIMEOUT_SECONDS) as response:
            return 200 <= response.getcode() < 300
    except (OSError, ValueError):
        return False


def _remove_pending_report(event_id, stats_path):
    try:
        with _usage_stats_lock(stats_path):
            stats = read_usage_stats(stats_path)
            before = len(stats["pending_reports"])
            stats["pending_reports"] = [
                report for report in stats["pending_reports"] if report["event_id"] != event_id
            ]
            return before != len(stats["pending_reports"]) and write_usage_stats(stats, stats_path)
    except Exception:  # noqa: BLE001 -- delivery failures stay queued
        return False


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def _usage_stats_lock(stats_path):
    """Serialize state updates across the app's supported windows."""
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stats_path.with_name(f".{stats_path.name}.lock")
    with _usage_stats_thread_lock:
        lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT)
        acquired = False
        try:
            deadline = time.monotonic() + USAGE_STATS_LOCK_TIMEOUT_SECONDS
            while not acquired:
                try:
                    _try_lock(lock_fd)
                    acquired = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(USAGE_STATS_LOCK_RETRY_SECONDS)
            yield
        finally:
            if acquired:
                _unlock(lock_fd)
            os.close(lock_fd)


def _try_lock(lock_fd):
    if "msvcrt" in globals():
        if os.fstat(lock_fd).st_size == 0:
            os.write(lock_fd, b"0")
        os.lseek(lock_fd, 0, os.SEEK_SET)
        msvcrt.locking(lock_fd, msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - exercised on macOS/Linux
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(lock_fd):
    if "msvcrt" in globals():
        os.lseek(lock_fd, 0, os.SEEK_SET)
        msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - exercised on macOS/Linux
        fcntl.flock(lock_fd, fcntl.LOCK_UN)


def write_usage_stats(stats, stats_file=None):
    """Replace state atomically. The surrounding lock prevents lost updates."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    temp_path = None
    try:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=stats_path.parent,
            prefix=f".{stats_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(stats, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, stats_path)
        return True
    except OSError:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except OSError:
                pass
        return False


def format_usage_summary(stats, app_version=""):
    """Return the shareable local summary; opaque IDs and fingerprints stay out."""
    hours_scored = stats["seconds_scored"] / SECONDS_PER_HOUR
    lines = [
        "Sleep Scoring App -- research impact summary",
        "",
        f"Recordings scored:    {stats['recordings_scored']}",
        f"Hours scored:         {hours_scored:.1f}",
        f"First scored on:      {stats['first_recorded_at'] or 'n/a'}",
        f"Most recently scored: {stats['last_recorded_at'] or 'n/a'}",
    ]
    if app_version:
        lines.append(f"App version:          {app_version}")
    lines += [
        "",
        "A recording is counted once, the first time it is saved with every",
        "second scored. This summary contains no app identifier, recording names,",
        "paths, signal data, annotations, or animal identifiers.",
    ]
    return "\n".join(lines)
