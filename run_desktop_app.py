# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 17:54:32 2025

@author: yzhao
"""

import multiprocessing
import os
import sys
import threading

import webview


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


def run_startup_update_if_enabled():
    if not should_run_startup_update():
        return

    try:
        from desktop_app_source_updater import (
            UpdateConfig,
            format_update_message,
            run_startup_update,
        )
    except ImportError as exc:
        print(f"[startup-update] updater unavailable: {exc}", flush=True)
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
        print(f"[startup-update] failed unexpectedly: {exc}", flush=True)
        return

    message = format_update_message(result)
    if message:
        print(f"[startup-update] {message}", flush=True)


def run_dash(app, port):
    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
        dev_tools_hot_reload=False,
        use_reloader=False,
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--smoke" not in argv:
        run_startup_update_if_enabled()

    from app_src import VERSION
    from app_src.app import app
    from app_src.config import PORT, WINDOW_CONFIG

    if "--smoke" in argv:
        print(f"Sleep Scoring App {VERSION} smoke check OK")
        return 0

    multiprocessing.freeze_support()
    t = threading.Thread(target=run_dash, args=(app, PORT), daemon=True)
    t.start()
    webview.settings["ALLOW_DOWNLOADS"] = True  # must have this for the download to work

    # This is the window `webview.windows[0]` will refer to.
    webview.create_window(
        f"Sleep Scoring App {VERSION}",
        f"http://127.0.0.1:{PORT}",
        **WINDOW_CONFIG,
    )

    # Start pywebview (Windows: force edgechromium; others: auto-select native renderer).
    if sys.platform == "win32":
        webview.start(gui="edgechromium")
    else:
        webview.start()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
