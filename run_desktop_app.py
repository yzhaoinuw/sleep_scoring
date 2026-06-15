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


def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.abspath(os.path.dirname(__file__))


base_path = get_base_path()

# Insert base_path first so app_src next to the executable stays patchable.
sys.path.insert(0, base_path)


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
