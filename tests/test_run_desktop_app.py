import socket
import sys
from types import SimpleNamespace

import run_desktop_app


def test_startup_update_prints_checking_and_result(monkeypatch, capsys):
    result = SimpleNamespace(status="up-to-date", message="installed version is current")
    fake_updater = SimpleNamespace(
        UpdateConfig=lambda **kwargs: kwargs,
        format_update_message=lambda update_result: "",
        run_startup_update=lambda config: result,
    )

    monkeypatch.setattr(run_desktop_app, "should_run_startup_update", lambda: True)
    monkeypatch.setitem(sys.modules, "desktop_app_source_updater", fake_updater)

    run_desktop_app.run_startup_update_if_enabled()

    assert capsys.readouterr().out.strip().splitlines() == [
        "[startup-update] checking for updates...",
        "[startup-update] no update available",
    ]


def test_source_run_prints_skipped_message(monkeypatch, capsys):
    monkeypatch.delenv(run_desktop_app.SKIP_UPDATE_ENV, raising=False)
    monkeypatch.delenv(run_desktop_app.UPDATE_ZIP_URL_ENV, raising=False)
    monkeypatch.delenv(run_desktop_app.UPDATE_RELEASE_API_ENV, raising=False)
    monkeypatch.setattr(run_desktop_app.sys, "frozen", False, raising=False)

    run_desktop_app.run_startup_update_if_enabled()

    assert capsys.readouterr().out.strip() == (
        "[startup-update] source run; automatic update check skipped"
    )


def test_skip_env_prints_disabled_message(monkeypatch, capsys):
    monkeypatch.setenv(run_desktop_app.SKIP_UPDATE_ENV, "1")

    run_desktop_app.run_startup_update_if_enabled()

    assert capsys.readouterr().out.strip() == "[startup-update] update check disabled"


def test_formats_successful_update_message():
    result = SimpleNamespace(status="updated", message="updated to v1.2.3")

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "updated to v1.2.3 (4 changed files)",
    )

    assert message == "updated to v1.2.3 (4 changed files)"


def test_formats_failed_update_message_as_non_blocking():
    result = SimpleNamespace(status="failed", message="could not download update metadata")

    message = run_desktop_app.format_startup_update_console_message(
        result, lambda update_result: ""
    )

    assert message == "update check failed: could not download update metadata; continuing startup"


def test_formats_skipped_update_message_as_non_blocking():
    result = SimpleNamespace(status="skipped", message="local runtime files differ")

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "local runtime files differ: app_src/app.py",
    )

    assert (
        message
        == "update not applied: local runtime files differ: app_src/app.py; continuing startup"
    )


def _bind_ephemeral_socket():
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    return holder, holder.getsockname()[1]


def test_claims_slot_zero_when_base_port_is_free():
    holder, port = _bind_ephemeral_socket()
    holder.close()  # freed port becomes the base of an all-free slot range

    slot, claimed_port, probe_socket = run_desktop_app.claim_session_slot(
        base_port=port, max_sessions=3
    )

    try:
        assert (slot, claimed_port) == (0, port)
        assert probe_socket.getsockname() == ("127.0.0.1", port)
    finally:
        probe_socket.close()


def test_skips_occupied_slot_and_claims_next():
    holder, base_port = _bind_ephemeral_socket()

    try:
        slot, claimed_port, probe_socket = run_desktop_app.claim_session_slot(
            base_port=base_port, max_sessions=3
        )
        try:
            assert (slot, claimed_port) == (1, base_port + 1)
        finally:
            probe_socket.close()
    finally:
        holder.close()


def test_returns_none_when_all_slots_are_taken():
    holder, base_port = _bind_ephemeral_socket()

    try:
        result = run_desktop_app.claim_session_slot(base_port=base_port, max_sessions=1)
    finally:
        holder.close()

    assert result == (None, None, None)
