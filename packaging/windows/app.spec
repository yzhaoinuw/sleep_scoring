# -*- mode: python ; coding: utf-8 -*-

from importlib.util import find_spec
import os
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(os.environ.get("SLEEP_SCORING_REPO_ROOT", Path.cwd())).resolve()
sys.path.insert(0, str(ROOT))

from app_src import VERSION  # noqa: E402


def package_dir(package_name):
    spec = find_spec(package_name)
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(f"Could not locate package {package_name!r} in the build environment.")
    return Path(next(iter(spec.submodule_search_locations))).resolve()


datas = [
    (str(package_dir("dash_player")), "dash_player"),
    (str(package_dir("dash_extensions")), "dash_extensions"),
    (str(ROOT / "app_src" / "assets"), "assets"),
    (str(package_dir("scipy")), "scipy"),
]
datas += collect_data_files("einops", include_py_files=True)

a = Analysis(
    [str(ROOT / "run_desktop_app.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch"],
    noarchive=False,
)

# Keep app_src patchable beside the executable instead of frozen into _internal.
a.pure = [item for item in a.pure if "app_src" not in item[0]]
a.scripts = [item for item in a.scripts if "app_src" not in item[0]]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="run_desktop_app",
    debug=False,
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
    name=f"sleep_scoring_app_{VERSION}",
)
