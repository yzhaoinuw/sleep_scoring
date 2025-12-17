# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_data_files

# Add the current working directory to sys.path
sys.path.insert(0, os.getcwd())

from app_src import VERSION


datas=[
    ('C:\\Users\\yzhao\\miniconda3\\envs\\sleep_scoring_dash3.0_dist\\lib\\site-packages\\dash_player', 'dash_player'),
    ('C:\\Users\\yzhao\\miniconda3\\envs\\sleep_scoring_dash3.0_dist\\lib\\site-packages\\dash_extensions', 'dash_extensions'),
    ('C:\\Users\\yzhao\\python_projects\\sleep_scoring\\app_src\\assets', 'assets'),
    ('C:\\Users\\yzhao\\miniconda3\\envs\\sleep_scoring_dash3.0_dist\\lib\\site-packages\\scipy', 'scipy'),
]
datas += collect_data_files('timm', include_py_files=True)

a = Analysis(
    ['run_desktop_app.py'],
    pathex=['C:\\Users\\yzhao\\miniconda3\\envs\\sleep_scoring_dash3.0_dist\\lib\\site-packages'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Remove app_src modules from the Analysis

a.pure = [x for x in a.pure if 'app_src' not in x[0]]
a.scripts = [x for x in a.scripts if 'app_src' not in x[0]]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run_desktop_app',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f'sleep_scoring_app_{VERSION}',
)
