"""Real MAT saves with injected I/O failures and recovery/cleanup checks."""

from unittest.mock import Mock

import numpy as np
import pytest
from scipy.io import loadmat, savemat

from app_src import atomic_save
from app_src.callbacks import saving


@pytest.fixture
def recording(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    source = shared / "recording.mat"
    mat = {
        "eeg": np.arange(512 * 5, dtype=np.float32),
        "emg": np.arange(512 * 5, dtype=np.float32) * 2,
        "eeg_frequency": 512.0,
        "sleep_scores": np.full(5, -1.0),
    }
    savemat(source, mat)
    labels = np.array([0, 1, np.nan, 2, np.nan])
    values = {"filepath": str(source), "filename": "recording", "sleep_scores_history": [labels]}
    monkeypatch.setattr(saving, "TEMP_PATH", local)
    monkeypatch.setattr(saving.cache, "get", values.get)
    count = Mock()
    monkeypatch.setattr(saving, "record_scored_recording", count)
    return source, local, shared, labels, values, count


@pytest.mark.parametrize("overwrite_source", [False, True])
def test_save_roundtrip_and_success_cleanup(recording, monkeypatch, overwrite_source):
    source, local, shared, labels, values, count = recording
    original = source.read_bytes()
    target = source if overwrite_source else shared / "recording_scored.mat"
    monkeypatch.setattr(saving, "save_file_dialog", lambda *args: str(target))
    message, intervals = saving.save_annotations(1)

    mat = loadmat(target, squeeze_me=True)
    np.testing.assert_array_equal(mat["sleep_scores"], [0, 1, -1, 2, -1])
    np.testing.assert_array_equal(mat["eeg"], np.arange(512 * 5, dtype=np.float32))
    np.testing.assert_array_equal(mat["emg"], np.arange(512 * 5, dtype=np.float32) * 2)
    np.testing.assert_array_equal(labels, [0, 1, np.nan, 2, np.nan])
    assert "Saved annotations to" in message
    assert "Unscored segment found: [2, 3]" in message
    assert intervals == 5
    assert not list(local.iterdir())
    assert not list(shared.glob(".*.tmp"))
    if not overwrite_source:
        assert source.read_bytes() == original
    count.assert_not_called()


@pytest.mark.parametrize("failure", ["copy", "verification", "replace"])
def test_failed_overwrite_keeps_original_and_complete_recovery(recording, monkeypatch, failure):
    source, local, shared, labels, values, count = recording
    original = source.read_bytes()
    values["sleep_scores_history"] = [np.zeros(5)]  # would normally count/export
    dialog = Mock(return_value=str(source))
    monkeypatch.setattr(saving, "save_file_dialog", dialog)

    def broken_copy(inp, out, **kwargs):
        out.write(inp.read(128))
        if failure == "copy":
            raise OSError("drive disconnected")

    if failure == "replace":
        monkeypatch.setattr(atomic_save.os, "replace", Mock(side_effect=PermissionError("locked")))
    else:
        monkeypatch.setattr(atomic_save.shutil, "copyfileobj", broken_copy)

    message, intervals = saving.save_annotations(1)

    assert "Could not save annotations" in message
    assert "Saved annotations to" not in message
    assert intervals == 0  # keep the recovery message visible
    assert source.read_bytes() == original
    exports = list(local.glob("*.mat"))
    assert len(exports) == 1
    assert str(exports[0]) in message
    np.testing.assert_array_equal(loadmat(exports[0], squeeze_me=True)["sleep_scores"], np.zeros(5))
    assert not list(shared.glob(".*.tmp"))
    assert not list(local.glob("*.xlsx"))
    dialog.assert_called_once()
    count.assert_not_called()


def test_cancelled_save_removes_temporary_mat(recording, monkeypatch):
    source, local, shared, labels, values, count = recording
    original = source.read_bytes()
    monkeypatch.setattr(saving, "save_file_dialog", lambda *args: None)
    message, _ = saving.save_annotations(1)
    assert "Unscored segment found" in message
    assert not list(local.iterdir())
    assert source.read_bytes() == original
    count.assert_not_called()


def test_failed_serialization_keeps_previous_saves_and_session_labels(recording, monkeypatch):
    source, local, shared, labels, values, count = recording
    original = source.read_bytes()
    previous_recovery = local / "previous_recovery.mat"
    previous_recovery.write_bytes(original)
    dialog = Mock()
    monkeypatch.setattr(saving, "save_file_dialog", dialog)

    def fail_savemat(export, mat):
        export.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(saving, "savemat", fail_savemat)
    message, intervals = saving.save_annotations(1)
    assert "Could not prepare" in message
    assert intervals == 0
    assert source.read_bytes() == original
    assert previous_recovery.read_bytes() == original
    assert list(local.iterdir()) == [previous_recovery]
    np.testing.assert_array_equal(labels, [0, 1, np.nan, 2, np.nan])
    dialog.assert_not_called()
    count.assert_not_called()


def test_selected_local_export_is_not_deleted(recording, monkeypatch):
    source, local, shared, labels, values, count = recording
    monkeypatch.setattr(saving, "save_file_dialog", lambda *args: str(next(local.glob("*.mat"))))
    message, _ = saving.save_annotations(1)
    exports = list(local.glob("*.mat"))
    assert len(exports) == 1
    assert str(exports[0]) in message
    np.testing.assert_array_equal(
        loadmat(exports[0], squeeze_me=True)["sleep_scores"], [0, 1, -1, 2, -1]
    )


def test_unreadable_source_reports_failure_without_exporting(recording, monkeypatch):
    source, local, shared, labels, values, count = recording
    source.write_bytes(b"truncated MAT")
    dialog = Mock()
    monkeypatch.setattr(saving, "save_file_dialog", dialog)
    message, intervals = saving.save_annotations(1)
    assert "Could not read the recording" in message
    assert intervals == 0
    assert not list(local.iterdir())
    dialog.assert_not_called()
    count.assert_not_called()
