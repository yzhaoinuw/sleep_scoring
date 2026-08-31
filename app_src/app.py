# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""


import threading

from app_src.server import app  # re-exported for run_desktop_app.py
from app_src.usage_stats import configure_usage_reporting, sync_usage_reports

# Importing these modules registers the Flask routes and Dash callbacks on the
# shared app instance.
import app_src.routes
import app_src.callbacks


def _sync_configured_usage_reports():
    """Retry configured opt-in reports without delaying the interactive app."""
    if configure_usage_reporting():
        sync_usage_reports()


# The hidden config opt-in creates an enrollment event once, then sends queued
# aggregate reports on future launches. Annotation saves never make a request.
threading.Thread(target=_sync_configured_usage_reports, daemon=True).start()
