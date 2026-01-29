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

if getattr(sys, "frozen", False):
    # Running as packaged .exe → base path is folder containing executable
    base_path = os.path.dirname(sys.executable)
else:
    # Running as normal script → base path is folder containing this file
    base_path = os.path.abspath(os.path.dirname(__file__))

# Insert base_path FIRST so that fp_analysis_app/ next to .exe overrides bundled version
sys.path.insert(0, base_path)


def run_dash():
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        dev_tools_hot_reload=False,
    )


if __name__ == "__main__":
    from app_src import VERSION
    from app_src.app_dev import app
    from app_src.config import PORT, WINDOW_CONFIG

    multiprocessing.freeze_support()
    t = threading.Thread(target=run_dash, daemon=True)
    t.start()
    webview.settings["ALLOW_DOWNLOADS"] = True  # must have this for the download to work
    # This is the window `webview.windows[0]` will refer to
    webview.create_window(
        f"Sleep Scoring App {VERSION}",
        f"http://127.0.0.1:{PORT}",
        **WINDOW_CONFIG,
    )

    # Start pywebview (Windows → force edgechromium, others → auto)
    if sys.platform == "win32":
        webview.start(gui="edgechromium")
    else:
        webview.start()  # macOS/Linux auto-selects native renderer
