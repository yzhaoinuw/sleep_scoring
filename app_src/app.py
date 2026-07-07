# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:45:29 2023

@author: yzhao
"""


from app_src.server import app  # re-exported for run_desktop_app.py

# Importing these modules registers the Flask routes and Dash callbacks on the
# shared app instance.
import app_src.routes
import app_src.callbacks
