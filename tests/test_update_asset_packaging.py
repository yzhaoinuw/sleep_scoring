import hashlib
import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "align_update_asset_with_package.py"
)
SPEC = importlib.util.spec_from_file_location("align_update_asset_with_package", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

EXPORT_SCRIPT_PATH = (
    Path(__file__).parents[1] / "packaging" / "windows" / "export_runtime_from_git.py"
)
EXPORT_SPEC = importlib.util.spec_from_file_location("export_runtime_from_git", EXPORT_SCRIPT_PATH)
EXPORT_MODULE = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(EXPORT_MODULE)


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
    assert set(manifest["files"][0]["previous_sha256"]) == {
        "git-blob-hash",
        hashlib.sha256(b"previous\r\n").hexdigest(),
    }
    assert "previous_sha256_by_version" not in manifest["files"][0]


def test_align_update_asset_preserves_multiple_package_lineages(tmp_path):
    update_zip = tmp_path / "update.zip"
    full_package_zip = tmp_path / "full.zip"
    patched_package_zip = tmp_path / "patched.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(full_package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("full/app_src/session.py", b"full-package\r\n")
    with zipfile.ZipFile(patched_package_zip, "w", zipfile.ZIP_DEFLATED) as package:
        package.writestr("patched/app_src/session.py", b"previous-source-update\n")

    MODULE.align_update_asset(
        update_zip,
        [
            ("v0.16.5", full_package_zip),
            ("v0.16.5", patched_package_zip),
        ],
    )

    with zipfile.ZipFile(update_zip) as update:
        manifest = json.loads(update.read("manifest.json"))
    assert set(manifest["files"][0]["previous_sha256"]) == {
        "git-blob-hash",
        hashlib.sha256(b"full-package\r\n").hexdigest(),
        hashlib.sha256(b"previous-source-update\n").hexdigest(),
    }


def test_align_update_asset_rejects_missing_packaged_runtime_file(tmp_path):
    update_zip = tmp_path / "update.zip"
    package_zip = tmp_path / "full.zip"
    _write_update_zip(update_zip)
    with zipfile.ZipFile(package_zip, "w"):
        pass

    with pytest.raises(ValueError, match="expected exactly one"):
        MODULE.align_update_asset(update_zip, [("v0.16.5", package_zip)])


def test_align_update_asset_omits_preserved_user_config(tmp_path):
    update_zip = tmp_path / "update.zip"
    manifest = {
        "from_versions": ["v0.16.6"],
        "changed_files": ["app_src/config.py", "app_src/make_figure.py"],
        "files": [
            {
                "path": "app_src/config.py",
                "sha256": "new-config",
                "previous_sha256_by_version": {"v0.16.6": "old-config"},
            },
            {
                "path": "app_src/make_figure.py",
                "sha256": "new-figure",
                "previous_sha256_by_version": {"v0.16.6": "old-figure"},
            },
        ],
    }
    with zipfile.ZipFile(update_zip, "w", zipfile.ZIP_DEFLATED) as update:
        update.writestr("manifest.json", json.dumps(manifest))
        update.writestr("app_src/config.py", b"new config\n")
        update.writestr("app_src/make_figure.py", b"new figure\n")

    MODULE.align_update_asset(update_zip, [], ("app_src/config.py",))

    with zipfile.ZipFile(update_zip) as update:
        updated_manifest = json.loads(update.read("manifest.json"))
        assert "app_src/config.py" not in update.namelist()
        assert update.read("app_src/make_figure.py") == b"new figure\n"
    assert updated_manifest["changed_files"] == ["app_src/make_figure.py"]
    assert [entry["path"] for entry in updated_manifest["files"]] == ["app_src/make_figure.py"]


def test_export_runtime_writes_exact_git_blob_bytes(tmp_path):
    repo = Path(__file__).parents[1]

    exported = EXPORT_MODULE.export_runtime(repo, "HEAD", "app_src", tmp_path)

    assert "app_src/__init__.py" in exported
    expected = subprocess.check_output(["git", "-C", str(repo), "show", "HEAD:app_src/__init__.py"])
    assert (tmp_path / "app_src" / "__init__.py").read_bytes() == expected
