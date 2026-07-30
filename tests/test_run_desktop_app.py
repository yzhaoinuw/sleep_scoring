import socket
import sys
from types import SimpleNamespace

import run_desktop_app


def test_reads_installed_version_without_importing_app_src(monkeypatch, tmp_path):
    app_src = tmp_path / "app_src"
    app_src.mkdir()
    (app_src / "__init__.py").write_text('VERSION = "v1.2.3"\n', encoding="utf-8")
    monkeypatch.setattr(run_desktop_app, "base_path", str(tmp_path))

    assert run_desktop_app.get_installed_version() == "v1.2.3"


def test_missing_installed_version_is_reported_as_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(run_desktop_app, "base_path", str(tmp_path))

    assert run_desktop_app.get_installed_version() == "unknown"


def test_startup_update_prints_checking_and_result(monkeypatch, capsys, tmp_path):
    result = SimpleNamespace(status="up-to-date", message="installed version is current")
    calls = []

    def fake_run_startup_update(config, *, force_check=False):
        calls.append((config, force_check))
        return result

    fake_updater = SimpleNamespace(
        UpdateConfig=lambda **kwargs: kwargs,
        format_update_message=lambda update_result: "",
        run_startup_update=fake_run_startup_update,
    )

    monkeypatch.setattr(run_desktop_app, "should_run_startup_update", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setitem(sys.modules, "desktop_app_source_updater", fake_updater)

    succeeded = run_desktop_app.run_startup_update_if_enabled(force_check=True)

    assert succeeded is True
    config, force_check = calls[0]
    assert force_check is True
    assert config["latest_release_url"] == run_desktop_app.LATEST_RELEASE_URL
    assert config["release_api_url"] == ""
    assert config["check_state_file"] == tmp_path / "sleep_scoring" / "update-check.json"
    assert config["force_check_env"] == run_desktop_app.FORCE_UPDATE_CHECK_ENV
    assert config["on_update_available"] is run_desktop_app.show_update_available
    assert capsys.readouterr().out.strip().splitlines() == [
        "[startup-update] checking for updates...",
        "[startup-update] no update available",
    ]


def test_source_run_prints_skipped_message(monkeypatch, capsys):
    monkeypatch.delenv(run_desktop_app.SKIP_UPDATE_ENV, raising=False)
    monkeypatch.delenv(run_desktop_app.UPDATE_ZIP_URL_ENV, raising=False)
    monkeypatch.delenv(run_desktop_app.UPDATE_LATEST_RELEASE_ENV, raising=False)
    monkeypatch.delenv(run_desktop_app.UPDATE_RELEASE_API_ENV, raising=False)
    monkeypatch.setattr(run_desktop_app.sys, "frozen", False, raising=False)

    succeeded = run_desktop_app.run_startup_update_if_enabled()

    assert succeeded is True
    assert capsys.readouterr().out.strip() == (
        "[startup-update] source run; automatic update check skipped"
    )


def test_skip_env_prints_disabled_message(monkeypatch, capsys):
    monkeypatch.setenv(run_desktop_app.SKIP_UPDATE_ENV, "1")

    succeeded = run_desktop_app.run_startup_update_if_enabled()

    assert succeeded is True
    assert capsys.readouterr().out.strip() == "[startup-update] update check disabled"


def test_update_check_mode_uses_failure_exit_code(monkeypatch):
    monkeypatch.setattr(run_desktop_app, "print_installed_version", lambda: None)
    received = []
    monkeypatch.setattr(
        run_desktop_app,
        "run_startup_update_if_enabled",
        lambda force_check=False: received.append(force_check) or False,
    )

    assert run_desktop_app.main(["--check-update"]) == 1
    assert received == [True]


def test_update_check_mode_uses_success_exit_code(monkeypatch):
    monkeypatch.setattr(run_desktop_app, "print_installed_version", lambda: None)
    received = []
    monkeypatch.setattr(
        run_desktop_app,
        "run_startup_update_if_enabled",
        lambda force_check=False: received.append(force_check) or True,
    )

    assert run_desktop_app.main(["--check-update"]) == 0
    assert received == [True]


def test_update_check_prints_version_before_checking(monkeypatch, capsys):
    monkeypatch.setattr(run_desktop_app, "get_installed_version", lambda: "v1.2.3")

    def fake_update(force_check=False):
        assert force_check is True
        print("[startup-update] checking for updates...")
        return True

    monkeypatch.setattr(run_desktop_app, "run_startup_update_if_enabled", fake_update)

    assert run_desktop_app.main(["--check-update"]) == 0
    assert capsys.readouterr().out.strip().splitlines() == [
        "[startup] Sleep Scoring App version: v1.2.3",
        "[startup-update] checking for updates...",
    ]


def test_formats_successful_update_message():
    result = SimpleNamespace(status="updated", message="updated to v1.2.3")

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "updated to v1.2.3 (4 changed files)",
    )

    assert message == "updated to v1.2.3 (4 changed files)"


def test_prints_explicit_update_versions(capsys):
    run_desktop_app.show_update_available("v1.2.3", "v1.3.0")

    assert capsys.readouterr().out.strip() == (
        "[startup-update] updating from version v1.2.3 to version v1.3.0..."
    )


def test_legacy_api_override_disables_default_redirect(monkeypatch):
    monkeypatch.setenv(
        run_desktop_app.UPDATE_RELEASE_API_ENV,
        "https://example.test/releases/latest",
    )
    monkeypatch.delenv(run_desktop_app.UPDATE_LATEST_RELEASE_ENV, raising=False)

    assert run_desktop_app.get_latest_release_url() == ""


def test_latest_release_override_takes_precedence_over_legacy_api(monkeypatch):
    monkeypatch.setenv(
        run_desktop_app.UPDATE_RELEASE_API_ENV,
        "https://example.test/api/releases/latest",
    )
    monkeypatch.setenv(
        run_desktop_app.UPDATE_LATEST_RELEASE_ENV,
        "https://example.test/releases/latest",
    )

    assert run_desktop_app.get_latest_release_url() == run_desktop_app.LATEST_RELEASE_URL


def test_update_state_file_can_be_overridden(monkeypatch, tmp_path):
    state_file = tmp_path / "custom-update-state.json"
    monkeypatch.setenv(run_desktop_app.UPDATE_STATE_FILE_ENV, str(state_file))

    assert run_desktop_app.get_update_state_file() == str(state_file)


def test_formats_deferred_check_as_network_skip():
    result = SimpleNamespace(
        status="up-to-date",
        message="startup update check deferred by the configured interval",
    )

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "",
    )

    assert message == "recent update check still current; network check skipped"


def test_formats_full_package_only_release_as_actionable():
    result = SimpleNamespace(
        status="up-to-date",
        message="release v0.18.0 has no matching source update asset",
        installed_version="0.17.0",
        target_version="v0.18.0",
    )

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "",
    )

    assert message == (
        "release v0.18.0 is available as a full package download; "
        "see the README installation steps"
    )


def test_current_release_is_still_reported_as_no_update():
    result = SimpleNamespace(
        status="up-to-date",
        message="installed version 0.17.0 is up to date",
        installed_version="0.17.0",
        target_version="v0.17.0",
    )

    message = run_desktop_app.format_startup_update_console_message(
        result,
        lambda update_result: "",
    )

    assert message == "no update available"


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


def _reserve_contiguous_port_range(count, attempts=20):
    """Bind `count` consecutive ports from an ephemeral base, so a test can
    reason about a slot range verified to be entirely free. Returns the base
    port and the bound holder sockets."""
    for _ in range(attempts):
        holder, base_port = _bind_ephemeral_socket()
        holders = [holder]
        try:
            for offset in range(1, count):
                extra = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                holders.append(extra)
                extra.bind(("127.0.0.1", base_port + offset))
        except OSError:
            for held in holders:
                held.close()
            continue
        return base_port, holders
    raise RuntimeError("could not reserve a contiguous free port range")


def test_no_peer_claim_when_a_peer_port_is_listening():
    # A live slot-1 window while this process holds slot 0: the launcher
    # must see the peer so it never patches app_src underneath it.
    listener, peer_port = _bind_ephemeral_socket()
    listener.listen(1)
    base_port = peer_port - 1

    try:
        assert run_desktop_app.claim_peer_slots(0, base_port=base_port, max_sessions=3) is None
    finally:
        listener.close()


def test_no_peer_claim_when_peer_port_bound_but_not_listening():
    # A peer that is still starting up holds its claimed port bound but not
    # yet listening until Dash takes over; a connect probe would miss it.
    holder, peer_port = _bind_ephemeral_socket()
    base_port = peer_port - 1

    try:
        assert run_desktop_app.claim_peer_slots(0, base_port=base_port, max_sessions=3) is None
    finally:
        holder.close()


def test_claims_and_holds_peer_ports_when_free():
    base_port, holders = _reserve_contiguous_port_range(3)
    own_claim, peer_holders = holders[0], holders[1:]
    for held in peer_holders:
        held.close()  # peers freed; own slot stays bound like a real claim

    guards = run_desktop_app.claim_peer_slots(0, base_port=base_port, max_sessions=3)

    try:
        assert [guard.getsockname()[1] for guard in guards] == [base_port + 1, base_port + 2]
    finally:
        for guard in guards:
            guard.close()
        own_claim.close()


def test_late_launcher_cannot_claim_peer_slot_until_guards_release():
    # A launcher starting while slot 0 still holds the update guards must
    # find every slot taken; releasing the guards reopens the slots.
    base_port, holders = _reserve_contiguous_port_range(3)
    own_claim, peer_holders = holders[0], holders[1:]
    for held in peer_holders:
        held.close()

    guards = run_desktop_app.claim_peer_slots(0, base_port=base_port, max_sessions=3)
    try:
        late = run_desktop_app.claim_session_slot(base_port=base_port, max_sessions=3)
        assert late == (None, None, None)
    finally:
        for guard in guards:
            guard.close()

    slot, claimed_port, probe_socket = run_desktop_app.claim_session_slot(
        base_port=base_port, max_sessions=3
    )
    try:
        assert (slot, claimed_port) == (1, base_port + 1)
    finally:
        probe_socket.close()
        own_claim.close()


def test_own_slot_listener_does_not_count_as_peer():
    # Only *other* slots matter; the claimed slot's own port is this process.
    listener, base_port = _bind_ephemeral_socket()
    listener.listen(1)

    try:
        assert run_desktop_app.claim_peer_slots(0, base_port=base_port, max_sessions=1) == []
    finally:
        listener.close()
