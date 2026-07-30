# -*- coding: utf-8 -*-
"""Local, aggregate-only record of app use for research impact reporting.

Nothing recorded here leaves the computer it runs on. There is no network
call in this module and no reporting endpoint anywhere in the app; the totals
exist so a user can export a summary and choose to share it.

The store holds counts and a set of opaque recording fingerprints. It never
holds file names, paths, signal values, annotations, or animal identifiers.
The fingerprints exist only so that reopening or re-saving the same recording
cannot inflate the totals. They are one-way digests of signal content, so they
cannot be turned back into a name or a path, but they are still stable
per-recording tokens: keep them local. Only the counts are meant to be shared.

A recording counts once, the first time it is saved with every second scored.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
USAGE_STATS_FILE_ENV = "SLEEP_SCORING_USAGE_STATS_FILE"
FINGERPRINT_LENGTH = 16
SECONDS_PER_HOUR = 3600


def get_usage_stats_file():
    """Return the per-user stats path, outside the app folder.

    A packaged app can be installed somewhere the user cannot write, and its
    folder is replaced wholesale by a full-package update, so the totals live
    beside the updater's own state instead.
    """
    configured_path = os.environ.get(USAGE_STATS_FILE_ENV)
    if configured_path:
        return Path(configured_path)
    state_root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache"))
    return state_root / "sleep_scoring" / "usage-stats.json"


def get_empty_usage_stats():
    return {
        "schema_version": SCHEMA_VERSION,
        "recordings_scored": 0,
        "seconds_scored": 0,
        "first_recorded_at": "",
        "last_recorded_at": "",
        "counted_recordings": [],
    }


def get_recording_fingerprint(mat):
    """Return a short one-way digest of a recording's EEG signal.

    The EEG array is the recording's identity: annotating and saving does not
    change it, so the same recording fingerprints identically no matter how
    many times it is reopened, re-saved, or saved under a new name.
    """
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
    """Return the stored totals, or empty totals when unreadable."""
    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    stats = get_empty_usage_stats()
    try:
        stored = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return stats

    if not isinstance(stored, dict) or stored.get("schema_version") != SCHEMA_VERSION:
        return stats

    for key, empty_value in stats.items():
        value = stored.get(key)
        if isinstance(value, type(empty_value)):
            stats[key] = value
    stats["counted_recordings"] = [str(fingerprint) for fingerprint in stats["counted_recordings"]]
    return stats


def record_scored_recording(mat, stats_file=None):
    """Count a fully scored recording once. Never raises.

    Returns True when this call added a recording to the totals. Usage
    accounting must not be able to break a save, so any failure here is
    swallowed: an uncounted recording is an acceptable loss, a lost save
    is not.
    """
    try:
        return _record_scored_recording(mat, stats_file)
    except Exception:  # noqa: BLE001 -- counting must never interrupt a save
        return False


def _record_scored_recording(mat, stats_file=None):
    fingerprint = get_recording_fingerprint(mat)
    scored_seconds = get_scored_seconds(mat)
    if not fingerprint or scored_seconds <= 0:
        return False

    stats_path = Path(stats_file) if stats_file is not None else get_usage_stats_file()
    stats = read_usage_stats(stats_path)
    if fingerprint in stats["counted_recordings"]:
        return False

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stats["recordings_scored"] += 1
    stats["seconds_scored"] += scored_seconds
    stats["counted_recordings"].append(fingerprint)
    stats["first_recorded_at"] = stats["first_recorded_at"] or now
    stats["last_recorded_at"] = now

    return write_usage_stats(stats, stats_path)


def write_usage_stats(stats, stats_file=None):
    """Replace the stats file atomically. Returns whether it was stored.

    The caller reports a recording as counted only when the totals actually
    reached disk; an unwritable store must not look like a successful count.

    Windows allows up to three app windows at once. Two saves completing in
    the same instant could lose one increment, since this is a read-modify-write
    with no lock. Saves are seconds apart at minimum and the cost is one
    uncounted recording, so the atomic replace is deliberately the only
    guarantee here; a lock file would add a failure mode worth more than the
    count it protects.
    """
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
    """Return the shareable summary: counts only, no fingerprints."""
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
        "second scored. These totals are stored only on this computer and are",
        "never transmitted anywhere. This summary contains no recording names,",
        "paths, signal data, annotations, or animal identifiers.",
    ]
    return "\n".join(lines)
