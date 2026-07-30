# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:54:32 2025

@author: yzhao
"""

import multiprocessing
import os
import re
import socket
import sys
import threading
from pathlib import Path

import webview


BASE_PORT = 8050
MAX_SESSIONS = 3
INSTANCE_SLOT_ENV = "SLEEP_SCORING_INSTANCE_SLOT"
PEER_PORTS_ENV = "SLEEP_SCORING_PEER_PORTS"

UPDATE_ASSET_PREFIX = "sleep_scoring_app_update_"
LATEST_RELEASE_URL = "https://github.com/yzhaoinuw/sleep_scoring/releases/latest"
SKIP_UPDATE_ENV = "SLEEP_SCORING_SKIP_UPDATE"
UPDATE_ZIP_URL_ENV = "SLEEP_SCORING_UPDATE_ZIP_URL"
UPDATE_LATEST_RELEASE_ENV = "SLEEP_SCORING_UPDATE_LATEST_RELEASE_URL"
UPDATE_RELEASE_API_ENV = "SLEEP_SCORING_UPDATE_RELEASE_API_URL"
UPDATE_ASSET_PREFIX_ENV = "SLEEP_SCORING_UPDATE_ASSET_PREFIX"
UPDATE_TIMEOUT_ENV = "SLEEP_SCORING_UPDATE_TIMEOUT_SECONDS"
UPDATE_STATE_FILE_ENV = "SLEEP_SCORING_UPDATE_STATE_FILE"
FORCE_UPDATE_CHECK_ENV = "SLEEP_SCORING_FORCE_UPDATE_CHECK"
VERSION_PATTERN = re.compile(r"""VERSION\s*=\s*["']([^"']+)["']""")


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))


base_path = get_base_path()

# Insert base_path first so app_src next to the executable stays patchable.
sys.path.insert(0, base_path)


def get_installed_version():
    version_file = os.path.join(base_path, "app_src", "__init__.py")
    try:
        with open(version_file, encoding="utf-8") as source:
            match = VERSION_PATTERN.search(source.read())
    except OSError:
        return "unknown"
    return match.group(1) if match else "unknown"


def print_installed_version():
    print(f"[startup] Sleep Scoring App version: {get_installed_version()}", flush=True)


def _env_flag_is_enabled(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def should_run_startup_update():
    if _env_flag_is_enabled(SKIP_UPDATE_ENV):
        return False
    if any(
        os.environ.get(name)
        for name in (
            UPDATE_ZIP_URL_ENV,
            UPDATE_LATEST_RELEASE_ENV,
            UPDATE_RELEASE_API_ENV,
        )
    ):
        return True
    return getattr(sys, "frozen", False)


def get_latest_release_url():
    if os.environ.get(UPDATE_RELEASE_API_ENV) and not os.environ.get(UPDATE_LATEST_RELEASE_ENV):
        return ""
    return LATEST_RELEASE_URL


def get_update_state_file():
    configured_path = os.environ.get(UPDATE_STATE_FILE_ENV)
    if configured_path:
        return configured_path
    state_root = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache"))
    return state_root / "sleep_scoring" / "update-check.json"


def show_update_available(installed_version, target_version):
    print(
        f"[startup-update] updating from version {installed_version} "
        f"to version {target_version}...",
        flush=True,
    )


def get_startup_update_skip_message():
    if _env_flag_is_enabled(SKIP_UPDATE_ENV):
        return "update check disabled"
    if not getattr(sys, "frozen", False):
        return "source run; automatic update check skipped"
    return ""


def run_startup_update_if_enabled(force_check=False):
    if not should_run_startup_update():
        message = get_startup_update_skip_message()
        if message:
            print(f"[startup-update] {message}", flush=True)
        return True

    print("[startup-update] checking for updates...", flush=True)

    try:
        from desktop_app_source_updater import (
            UpdateConfig,
            format_update_message,
            run_startup_update,
        )
    except ImportError as exc:
        print(f"[startup-update] updater unavailable: {exc}; continuing startup", flush=True)
        return False

    try:
        result = run_startup_update(
            UpdateConfig(
                app_name="sleep_scoring",
                app_root=base_path,
                installed_version_file="app_src/__init__.py",
                latest_release_url=get_latest_release_url(),
                latest_release_env=UPDATE_LATEST_RELEASE_ENV,
                release_api_url="",
                asset_prefix=UPDATE_ASSET_PREFIX,
                allowed_payload_paths=("app_src/",),
                check_state_file=get_update_state_file(),
                force_check_env=FORCE_UPDATE_CHECK_ENV,
                on_update_available=show_update_available,
                skip_update_env=SKIP_UPDATE_ENV,
                update_zip_url_env=UPDATE_ZIP_URL_ENV,
                release_api_env=UPDATE_RELEASE_API_ENV,
                asset_prefix_env=UPDATE_ASSET_PREFIX_ENV,
                timeout_env=UPDATE_TIMEOUT_ENV,
            ),
            force_check=force_check,
        )
    except Exception as exc:  # Keep update failures from blocking normal app startup.
        print(f"[startup-update] failed unexpectedly: {exc}; continuing startup", flush=True)
        return False

    message = format_startup_update_console_message(result, format_update_message)
    if message:
        print(f"[startup-update] {message}", flush=True)
    return result.status != "failed"


def get_version_sort_key(version):
    return tuple(int(part) for part in re.findall(r"\d+", version)) or (0,)


def found_release_needing_full_package(result):
    """True when the check found a newer release it cannot install itself.

    A release that ships no code-only update asset is a normal `up-to-date`
    outcome: full packages, and any later change to the update asset naming
    convention, look the same to an installed app. Reporting "no update
    available" would be wrong, because a newer release does exist and only
    needs a manual full-package install. Compare the versions rather than
    matching the updater's wording, since this launcher ships frozen and
    cannot be corrected by a source update.
    """
    installed_version = getattr(result, "installed_version", None)
    target_version = getattr(result, "target_version", None)
    if not installed_version or not target_version:
        return False
    return get_version_sort_key(target_version) > get_version_sort_key(installed_version)


def format_startup_update_console_message(result, format_update_message):
    message = format_update_message(result)
    if result.status == "up-to-date":
        if "deferred by the configured interval" in result.message:
            return "recent update check still current; network check skipped"
        if found_release_needing_full_package(result):
            return (
                f"release {result.target_version} is available as a full package "
                "download; see the README installation steps"
            )
        return "no update available"
    if result.status == "updated":
        return message or result.message
    if result.status == "failed":
        return f"update check failed: {result.message}; continuing startup"
    if result.status in {"blocked", "skipped"}:
        return f"update not applied: {message or result.message}; continuing startup"
    if result.status == "disabled":
        return "update check disabled"
    return message or result.message


def claim_session_slot(base_port=BASE_PORT, max_sessions=MAX_SESSIONS):
    """Bind the first free port in the slot range and return (slot, port, socket).

    The returned socket holds the claim until the Dash server takes the port
    over; keeping it bound closes the gap in which a concurrently launching
    window could scan its way onto the same slot. Returns (None, None, None)
    when every slot is taken.
    """
    for slot in range(max_sessions):
        port = base_port + slot
        probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe_socket.bind(("127.0.0.1", port))
        except OSError:
            probe_socket.close()
            continue
        return slot, port, probe_socket
    return None, None, None


def claim_peer_slots(slot, base_port=BASE_PORT, max_sessions=MAX_SESSIONS):
    """Bind and hold every other slot's port; None when any is already taken.

    Claiming slot 0 does not prove this is the only window: if the original
    slot-0 window closed while slots 1-2 stayed open, a relaunch reclaims
    slot 0 with live peers still running from app_src. The claim is a bind
    rather than a connect probe, because a peer that is still starting up
    holds its claimed port with a bound socket that is not listening yet
    and would refuse a connection. Anything holding a peer port counts as
    occupied; a non-app holder already blocks that slot for new windows,
    and skipping the update is the safe direction.

    The caller keeps the returned sockets bound for the whole update and
    closes them afterward: releasing them any earlier would let a launcher
    that starts mid-update claim a slot and import app_src while it is
    being patched. Until then, such a launcher sees every slot taken.
    """
    guards = []
    for other in range(max_sessions):
        if other == slot:
            continue
        guard = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            guard.bind(("127.0.0.1", base_port + other))
        except OSError:
            guard.close()
            for held in guards:
                held.close()
            return None
        guards.append(guard)
    return guards


def start_webview():
    # Windows: force edgechromium; others: auto-select native renderer.
    if sys.platform == "win32":
        webview.start(gui="edgechromium")
    else:
        webview.start()


def show_session_limit_message(max_sessions=MAX_SESSIONS):
    message = (
        f"{max_sessions} Sleep Scoring App windows are already open. "
        "Close one of them, then launch the app again."
    )
    print(f"[startup] {message}", flush=True)
    webview.create_window(
        "Sleep Scoring App",
        html=f"<p style='font-family: sans-serif; margin: 2em;'>{message}</p>",
        width=480,
        height=200,
        resizable=False,
    )
    start_webview()


def run_dash(app, port, probe_socket=None):
    if probe_socket is not None:
        probe_socket.close()  # release the claimed port just before Dash binds it
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        dev_tools_hot_reload=False,
        use_reloader=False,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--check-update" in argv:
        print_installed_version()
        return 0 if run_startup_update_if_enabled(force_check=True) else 1

    if "--smoke" in argv:
        from app_src import VERSION
        from app_src.app import app  # noqa: F401 -- importing the app is the check

        print(f"Sleep Scoring App {VERSION} smoke check OK")
        return 0

    slot, port, probe_socket = claim_session_slot()
    if slot is None:
        show_session_limit_message()
        return 1

    # Export the slot before importing app_src: server.py derives the
    # per-window temp/cache/video dirs from it at import time, and
    # loading.py uses the peer ports for the same-file check.
    os.environ[INSTANCE_SLOT_ENV] = str(slot)
    os.environ[PEER_PORTS_ENV] = ",".join(
        str(BASE_PORT + other) for other in range(MAX_SESSIONS) if other != slot
    )

    print_installed_version()
    peer_guards = claim_peer_slots(slot) if slot == 0 else None
    if peer_guards is None:
        # Never patch app_src while another window may be running from it.
        print(
            "[startup-update] another app window is running; update check skipped",
            flush=True,
        )
    else:
        try:
            run_startup_update_if_enabled()
        finally:
            for guard in peer_guards:
                guard.close()

    from app_src import VERSION
    from app_src.app import app
    from app_src.config import WINDOW_CONFIG

    multiprocessing.freeze_support()
    t = threading.Thread(target=run_dash, args=(app, port, probe_socket), daemon=True)
    t.start()
    webview.settings["ALLOW_DOWNLOADS"] = True  # must have this for the download to work

    window_title = f"Sleep Scoring App {VERSION}"
    if slot > 0:
        window_title += f" ({slot + 1})"

    # This is the window `webview.windows[0]` will refer to.
    webview.create_window(
        window_title,
        f"http://127.0.0.1:{port}",
        **WINDOW_CONFIG,
    )

    start_webview()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
