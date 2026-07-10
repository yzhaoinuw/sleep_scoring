# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:54:32 2025

@author: yzhao
"""

import multiprocessing
import os
import socket
import sys
import threading

import webview


BASE_PORT = 8050
MAX_SESSIONS = 3
INSTANCE_SLOT_ENV = "SLEEP_SCORING_INSTANCE_SLOT"
PEER_PORTS_ENV = "SLEEP_SCORING_PEER_PORTS"

UPDATE_ASSET_PREFIX = "sleep_scoring_app_update_"
SKIP_UPDATE_ENV = "SLEEP_SCORING_SKIP_UPDATE"
UPDATE_ZIP_URL_ENV = "SLEEP_SCORING_UPDATE_ZIP_URL"
UPDATE_RELEASE_API_ENV = "SLEEP_SCORING_UPDATE_RELEASE_API_URL"
UPDATE_ASSET_PREFIX_ENV = "SLEEP_SCORING_UPDATE_ASSET_PREFIX"
UPDATE_TIMEOUT_ENV = "SLEEP_SCORING_UPDATE_TIMEOUT_SECONDS"


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))


base_path = get_base_path()

# Insert base_path first so app_src next to the executable stays patchable.
sys.path.insert(0, base_path)


def _env_flag_is_enabled(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def should_run_startup_update():
    if _env_flag_is_enabled(SKIP_UPDATE_ENV):
        return False
    if os.environ.get(UPDATE_ZIP_URL_ENV) or os.environ.get(UPDATE_RELEASE_API_ENV):
        return True
    return getattr(sys, "frozen", False)


def get_startup_update_skip_message():
    if _env_flag_is_enabled(SKIP_UPDATE_ENV):
        return "update check disabled"
    if not getattr(sys, "frozen", False):
        return "source run; automatic update check skipped"
    return ""


def run_startup_update_if_enabled():
    if not should_run_startup_update():
        message = get_startup_update_skip_message()
        if message:
            print(f"[startup-update] {message}", flush=True)
        return

    print("[startup-update] checking for updates...", flush=True)

    try:
        from desktop_app_source_updater import (
            UpdateConfig,
            format_update_message,
            run_startup_update,
        )
    except ImportError as exc:
        print(f"[startup-update] updater unavailable: {exc}; continuing startup", flush=True)
        return

    try:
        result = run_startup_update(
            UpdateConfig(
                app_name="sleep_scoring",
                app_root=base_path,
                installed_version_file="app_src/__init__.py",
                release_api_url="https://api.github.com/repos/yzhaoinuw/sleep_scoring/releases/latest",
                asset_prefix=UPDATE_ASSET_PREFIX,
                allowed_payload_paths=("app_src/",),
                skip_update_env=SKIP_UPDATE_ENV,
                update_zip_url_env=UPDATE_ZIP_URL_ENV,
                release_api_env=UPDATE_RELEASE_API_ENV,
                asset_prefix_env=UPDATE_ASSET_PREFIX_ENV,
                timeout_env=UPDATE_TIMEOUT_ENV,
            )
        )
    except Exception as exc:  # Keep update failures from blocking normal app startup.
        print(f"[startup-update] failed unexpectedly: {exc}; continuing startup", flush=True)
        return

    message = format_startup_update_console_message(result, format_update_message)
    if message:
        print(f"[startup-update] {message}", flush=True)


def format_startup_update_console_message(result, format_update_message):
    message = format_update_message(result)
    if result.status == "up-to-date":
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


PEER_PROBE_TIMEOUT_SECONDS = 0.5


def any_peer_slot_occupied(slot, base_port=BASE_PORT, max_sessions=MAX_SESSIONS):
    """Return True when any other slot's port accepts a TCP connection.

    Claiming slot 0 does not prove this is the only window: if the original
    slot-0 window closed while slots 1-2 stayed open, a relaunch reclaims
    slot 0 with live peers still running from app_src. Any listener on a
    peer port counts as occupied; a non-app listener already blocks that
    slot for new windows, and skipping the update is the safe direction.
    """
    for other in range(max_sessions):
        if other == slot:
            continue
        try:
            probe = socket.create_connection(
                ("127.0.0.1", base_port + other), timeout=PEER_PROBE_TIMEOUT_SECONDS
            )
        except OSError:
            continue
        probe.close()
        return True
    return False


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

    if slot == 0 and not any_peer_slot_occupied(slot):
        run_startup_update_if_enabled()
    else:
        # Never patch app_src while another window may be running from it.
        print(
            "[startup-update] another app window is running; update check skipped",
            flush=True,
        )

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
