import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "align_update_asset_with_package.py"
)
SPEC = importlib.util.spec_from_file_location("align_update_asset_with_package", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_update_zip(path):
    manifest = {
        "from_versions": ["v0.16.5"],
        "files": [
            {
                "path": "app_src/session.py",
                "sha256": hashlib.sha256(b"new\n").hexdigest(),
                "previous_sha256_by_version": {"v0.16.5": "git-blob-hash"},
            }
        ],
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        update.writestr("app_src/session.py", b"new\n")


def test_align_update_asset_uses_exact_packaged_baseline_bytes(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("sleep_scoring_app_v0.16.5/app_src/session.py", b"previous\r\n")

    MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])

    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
        assert update.read("app_src/session.py") == b"new\n"
    assert manifest["files"][0]["previous_sha256_by_version"]["v0.16.5"] == (
        hashlib.sha256(b"previous\r\n").hexdigest()
    )


def test_align_update_asset_rejects_missing_packaged_runtime_file(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w"):
        pass

    with pytest.raises(ValueError, match="expected exactly one"):
        MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])
