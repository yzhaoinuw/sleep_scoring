# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""


import threading

from app_src.server import app  # re-exported for run_desktop_app.py
from app_src.usage_stats import sync_usage_reports

# Importing these modules registers the Flask routes and Dash callbacks on the
# shared app instance.
import app_src.routes
import app_src.callbacks

# Opted-in reports retry at launch without delaying the interactive app if a
# configured server is temporarily unavailable.
threading.Thread(target=sync_usage_reports, daemon=True).start()
