"""Failure paths must preserve an existing destination and the recovery export."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from app_src import atomic_save


@pytest.mark.parametrize("existing", [False, True])
def test_verified_save_replaces_only_after_copy_and_leaves_one_file(
    tmp_path, monkeypatch, existing
):
    source = tmp_path / "local" / "export.mat"
    source.parent.mkdir()
    source.write_bytes(b"complete new export" * 100)
    destination = tmp_path / "shared" / "recording.mat"
    destination.parent.mkdir()
    if existing:
        destination.write_bytes(b"previous good save")
    real_replace = atomic_save.os.replace

    def check_before_replace(staged, final):
        assert Path(staged).parent == destination.parent
        assert Path(staged).read_bytes() == source.read_bytes()
        assert Path(final) == destination
        if existing:
            assert destination.read_bytes() == b"previous good save"
        else:
            assert not destination.exists()
        real_replace(staged, final)

    monkeypatch.setattr(atomic_save.os, "replace", check_before_replace)
    atomic_save.copy_verified_export(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["copy", "short_copy", "corrupt_copy", "flush", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_failed_save_does_not_change_destination(tmp_path, monkeypatch, failure, existing):
    source = tmp_path / "export.mat"
    source.write_bytes(b"complete export")
    destination = tmp_path / "recording.mat"
    if existing:
        destination.write_bytes(b"previous good save")

    def broken_copy(original, staging, **kwargs):
        if failure == "corrupt_copy":
            staging.write(b"X" * len(source.read_bytes()))
        else:
            staging.write(original.read(3))
            if failure == "copy":
                raise OSError("drive disconnected")

    def fail(*args):
        raise PermissionError("simulated filesystem failure")

    if failure in {"copy", "short_copy", "corrupt_copy"}:
        monkeypatch.setattr(atomic_save.shutil, "copyfileobj", broken_copy)
    elif failure == "flush":
        monkeypatch.setattr(atomic_save.os, "fsync", fail)
    else:
        monkeypatch.setattr(atomic_save.os, "replace", fail)

    with pytest.raises(OSError):
        atomic_save.copy_verified_export(source, destination)

    if existing:
        assert destination.read_bytes() == b"previous good save"
    else:
        assert not destination.exists()
    assert source.read_bytes() == b"complete export"
    assert not list(tmp_path.glob(".*.tmp"))


def test_disconnected_destination_does_not_change_recovery_export(tmp_path):
    source = tmp_path / "export.mat"
    source.write_bytes(b"complete export")
    with pytest.raises(OSError):
        atomic_save.copy_verified_export(source, tmp_path / "unavailable" / "recording.mat")
    assert source.read_bytes() == b"complete export"


def test_process_termination_during_copy_preserves_previous_save(tmp_path):
    source = tmp_path / "export.mat"
    source.write_bytes(b"complete export" * 1000)
    destination = tmp_path / "recording.mat"
    destination.write_bytes(b"previous good save")
    ready = tmp_path / "copy_started"
    code = """
import sys, time
from pathlib import Path
from app_src import atomic_save
def stalled_copy(original, staging, **kwargs):
    staging.write(original.read(32))
    staging.flush()
    Path(sys.argv[3]).touch()
    time.sleep(60)
atomic_save.shutil.copyfileobj = stalled_copy
atomic_save.copy_verified_export(sys.argv[1], sys.argv[2])
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(source), str(destination), str(ready)],
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "Subprocess did not reach the staging copy"
        process.terminate()
        process.wait(timeout=10)
        assert destination.read_bytes() == b"previous good save"
        assert source.read_bytes() == b"complete export" * 1000
        # Forced termination cannot run cleanup, but the partial file has only
        # a staging name; it has never replaced the user's recording.
        assert len(list(tmp_path.glob(".*.tmp"))) == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
