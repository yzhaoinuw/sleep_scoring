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
